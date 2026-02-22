import asyncio
import base64
import concurrent.futures
import io
import random

import numpy as np
import socketio
from PIL import Image

from game_state import (
    generate_room_code,
    make_player,
    make_room,
    make_round,
    rooms,
    sid_to_player,
    sid_to_room,
)
from object_bank import OBJECTS, get_all_ids, get_object

# detector is injected by main.py after CLIPDetector is initialized
detector = None

# Single-threaded executor for CLIP inference (keeps GPU/MPS safe)
clip_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
)

# Per-player frame queues and worker tasks (keyed by socket id)
player_queues: dict[str, asyncio.Queue] = {}
player_workers: dict[str, asyncio.Task] = {}


# ─── helpers ─────────────────────────────────────────────────────────────────

def _scores(room: dict) -> dict:
    return {p["id"]: p["score"] for p in room["players"]}


async def _enqueue_frame(sid: str, frame_rgb: np.ndarray):
    """Put a frame in the player's queue, dropping stale frames to stay fresh."""
    q = player_queues.get(sid)
    if not q:
        return
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    await q.put(frame_rgb)


async def _start_player_worker(sid: str, room_code: str):
    if sid in player_workers:
        return
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    player_queues[sid] = q
    player_workers[sid] = asyncio.create_task(_frame_worker(sid, room_code))


async def _stop_player_worker(sid: str):
    task = player_workers.pop(sid, None)
    if task:
        task.cancel()
    player_queues.pop(sid, None)


async def _stop_all_workers(room: dict):
    for player in room["players"]:
        for s, pid in list(sid_to_player.items()):
            if pid == player["id"]:
                await _stop_player_worker(s)


async def _frame_worker(sid: str, room_code: str):
    """
    Per-player async worker.
    Pulls frames from the player's queue, runs CLIP in an executor thread,
    sends feedback to the player, and triggers round win when detected.
    """
    loop = asyncio.get_event_loop()
    q = player_queues[sid]

    while True:
        try:
            frame_rgb = await asyncio.wait_for(q.get(), timeout=10.0)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break

        try:
            result = await loop.run_in_executor(
                clip_executor,
                detector.detect_for_active_object,
                frame_rgb,
            )
        except Exception as e:
            print(f"[clip] worker error for {sid}: {e}")
            continue

        # Send live confidence feedback to this player only
        await sio.emit("frame:result", result, to=sid)

        if result["label"] != "none":
            await _handle_round_win(sid, room_code)


# ─── game flow ────────────────────────────────────────────────────────────────

async def _start_next_round(room_code: str):
    room = rooms.get(room_code)
    if not room or room["phase"] != "game":
        return

    used = set(room["usedObjectIds"])
    available = [oid for oid in get_all_ids() if oid not in used]

    if not available:
        # All objects used — reset pool
        used = set()
        available = get_all_ids()

    object_id = random.choice(available)
    used.add(object_id)
    room["usedObjectIds"] = list(used)

    obj = get_object(object_id)
    round_num = room["roundNumber"] + 1
    room["roundNumber"] = round_num

    rnd = make_round(str(round_num), object_id, obj["displayName"])
    room["currentRound"] = rnd

    # Pre-encode text embeddings for this object (blocking, but fast ~5ms)
    if detector:
        detector.set_active_object(object_id, obj)

    timeout = room["settings"]["roundTimeout"]
    timer_task = asyncio.create_task(
        _run_round_timeout(room_code, rnd["id"], timeout)
    )
    room["roundTimerTask"] = timer_task

    await sio.emit(
        "round:start",
        {
            "roundNumber": round_num,
            "objectId": object_id,
            "displayName": obj["displayName"],
            "timeoutSeconds": timeout,
            "players": room["players"],
            "scores": _scores(room),
        },
        room=room_code,
    )
    print(f"[game] round {round_num} → {obj['displayName']} in {room_code}")


async def _handle_round_win(sid: str, room_code: str):
    room = rooms.get(room_code)
    if not room or room["phase"] != "game":
        return

    current_round = room.get("currentRound")
    if not current_round or current_round.get("winnerId"):
        return  # Already won or no active round

    player_id = sid_to_player.get(sid)
    if not player_id:
        return

    # Mark winner (prevents double-win from concurrent detections)
    current_round["winnerId"] = player_id

    winner = None
    for p in room["players"]:
        if p["id"] == player_id:
            p["score"] += 1
            winner = p
            break

    if not winner:
        return

    current_round["winnerName"] = winner["name"]

    # Cancel the round timeout timer
    timer_task = room.get("roundTimerTask")
    if timer_task:
        timer_task.cancel()

    print(f"[game] round won by {winner['name']} in {room_code} (score: {winner['score']})")

    await sio.emit(
        "round:won",
        {
            "winnerId": player_id,
            "winnerName": winner["name"],
            "objectId": current_round["objectId"],
            "displayName": current_round["displayName"],
            "players": room["players"],
            "scores": _scores(room),
        },
        room=room_code,
    )

    # Check win condition
    points_to_win = room["settings"]["pointsToWin"]
    if winner["score"] >= points_to_win:
        await asyncio.sleep(3)
        await _end_game(room_code, player_id, winner["name"])
        return

    # Brief pause then next round
    await asyncio.sleep(3)
    await _start_next_round(room_code)


async def _run_round_timeout(room_code: str, round_id: str, timeout_seconds: int):
    await asyncio.sleep(timeout_seconds)

    room = rooms.get(room_code)
    if not room or room["phase"] != "game":
        return

    current_round = room.get("currentRound")
    if not current_round or current_round["id"] != round_id:
        return  # Round already finished

    if current_round.get("winnerId"):
        return  # Already won

    print(f"[game] round {round_id} timed out in {room_code}")

    await sio.emit(
        "round:timeout",
        {
            "objectId": current_round["objectId"],
            "displayName": current_round["displayName"],
            "scores": _scores(room),
        },
        room=room_code,
    )

    await asyncio.sleep(3)
    await _start_next_round(room_code)


async def _end_game(room_code: str, winner_id: str, winner_name: str):
    room = rooms.get(room_code)
    if not room:
        return

    room["phase"] = "results"

    timer_task = room.get("roundTimerTask")
    if timer_task:
        timer_task.cancel()

    await _stop_all_workers(room)

    await sio.emit(
        "game:ended",
        {
            "winnerId": winner_id,
            "winnerName": winner_name,
            "players": room["players"],
            "scores": _scores(room),
        },
        room=room_code,
    )
    print(f"[game] game:ended → {room_code}, winner: {winner_name}")


# ─── connection events ────────────────────────────────────────────────────────

@sio.event
async def connect(sid, environ):
    print(f"[socket] connect {sid}")


@sio.event
async def disconnect(sid):
    print(f"[socket] disconnect {sid}")

    await _stop_player_worker(sid)

    code = sid_to_room.pop(sid, None)
    player_id = sid_to_player.pop(sid, None)

    if not code or code not in rooms:
        return

    room = rooms[code]
    room["players"] = [p for p in room["players"] if p["id"] != player_id]

    if not room["players"]:
        del rooms[code]
        return

    if room["hostSid"] == sid:
        new_host = room["players"][0]
        room["hostId"] = new_host["id"]
        for s, c in list(sid_to_room.items()):
            if c == code and sid_to_player.get(s) == new_host["id"]:
                room["hostSid"] = s
                break

    await sio.emit("player:left", player_id, room=code)


# ─── room management ──────────────────────────────────────────────────────────

@sio.on("room:create")
async def room_create(sid, player_name: str):
    code = generate_room_code()
    while code in rooms:
        code = generate_room_code()

    player = make_player(player_name)
    room = make_room(code, player, sid)
    rooms[code] = room

    sid_to_room[sid] = code
    sid_to_player[sid] = player["id"]

    await sio.enter_room(sid, code)

    await sio.emit("room:created", {"roomCode": code}, to=sid)
    await sio.emit(
        "room:joined",
        {
            "players": room["players"],
            "settings": room["settings"],
            "playerId": player["id"],
            "hostId": player["id"],
        },
        to=sid,
    )
    print(f"[socket] room:create → {code} by {player['name']}")


@sio.on("room:join")
async def room_join(sid, data: dict):
    code = data.get("roomCode", "").strip().upper()
    player_name = data.get("playerName", "").strip()

    if code not in rooms:
        await sio.emit("error", {"message": "Room not found"}, to=sid)
        return

    room = rooms[code]

    if room["phase"] != "lobby":
        await sio.emit("error", {"message": "Game already in progress"}, to=sid)
        return

    player = make_player(player_name)
    room["players"].append(player)

    sid_to_room[sid] = code
    sid_to_player[sid] = player["id"]

    await sio.enter_room(sid, code)

    await sio.emit(
        "room:joined",
        {
            "players": room["players"],
            "settings": room["settings"],
            "playerId": player["id"],
            "hostId": room["hostId"],
        },
        to=sid,
    )
    await sio.emit("player:joined", player, room=code, skip_sid=sid)
    print(f"[socket] room:join → {code} by {player['name']}")


@sio.on("player:remove")
async def player_remove(sid, player_id: str):
    code = sid_to_room.get(sid)
    room = rooms.get(code)
    if not room or room["hostSid"] != sid:
        return
    room["players"] = [p for p in room["players"] if p["id"] != player_id]
    await sio.emit("player:left", player_id, room=code)


@sio.on("settings:update")
async def settings_update(sid, settings: dict):
    code = sid_to_room.get(sid)
    room = rooms.get(code)
    if not room or room["hostSid"] != sid:
        return

    allowed_points = [3, 5, 10, 15]
    allowed_timeout = [30, 60, 90, 120]

    if "pointsToWin" in settings and settings["pointsToWin"] in allowed_points:
        room["settings"]["pointsToWin"] = settings["pointsToWin"]
    if "roundTimeout" in settings and settings["roundTimeout"] in allowed_timeout:
        room["settings"]["roundTimeout"] = settings["roundTimeout"]

    await sio.emit("settings:updated", room["settings"], room=code)


# ─── game flow events ─────────────────────────────────────────────────────────

@sio.on("game:start")
async def game_start(sid):
    code = sid_to_room.get(sid)
    room = rooms.get(code)
    if not room or room["hostSid"] != sid:
        return
    if len(room["players"]) < 2:
        await sio.emit("error", {"message": "Need at least 2 players to start"}, to=sid)
        return
    if room["phase"] != "lobby":
        return

    room["phase"] = "game"
    room["roundNumber"] = 0
    room["usedObjectIds"] = []
    room["currentRound"] = None

    # Start a frame worker for every player in the room
    for player in room["players"]:
        for s, pid in list(sid_to_player.items()):
            if pid == player["id"]:
                await _start_player_worker(s, code)
                break

    await sio.emit(
        "game:started",
        {"players": room["players"], "settings": room["settings"]},
        room=code,
    )
    print(f"[socket] game:start → {code}")

    # Short countdown then kick off first round
    await asyncio.sleep(1)
    await _start_next_round(code)


# ─── frame submission ─────────────────────────────────────────────────────────

@sio.on("frame:submit")
async def frame_submit(sid, data: dict):
    code = sid_to_room.get(sid)
    room = rooms.get(code)
    if not room or room["phase"] != "game":
        return

    current_round = room.get("currentRound")
    if not current_round or current_round.get("winnerId"):
        return  # No active round or already won

    frame_b64: str = data.get("frameData", "")
    if not frame_b64:
        return

    # Decode base64 JPEG → RGB numpy array
    try:
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(frame_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # Resize to 224×224 — CLIP's native input size, reduces payload processing time
        pil_img = pil_img.resize((224, 224), Image.BILINEAR)
        np_img = np.array(pil_img)
    except Exception as e:
        print(f"[frame] decode error for {sid}: {e}")
        return

    await _enqueue_frame(sid, np_img)
