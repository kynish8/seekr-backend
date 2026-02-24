import asyncio
import base64
import concurrent.futures
import io
import os
import random

import numpy as np
import socketio
from PIL import Image

import redis_state
from game_state import (
    generate_room_code,
    make_player,
    make_room,
    make_round,
)
from object_bank import get_all_ids, get_object

# detector is injected by main.py after CLIPDetector is initialized
detector = None

# Single-threaded executor — keeps GPU/MPS inference serialised
clip_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Socket.io with Redis pub/sub manager — events work across multiple uvicorn workers
mgr = socketio.AsyncRedisManager(REDIS_URL)
sio = socketio.AsyncServer(
    client_manager=mgr,
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
)

# ── Process-local state (non-serialisable / per-worker ephemeral) ─────────────
# These live in memory only; Redis holds the canonical room/player data.
player_queues: dict[str, asyncio.Queue] = {}
player_workers: dict[str, asyncio.Task] = {}
_room_timer_tasks: dict[str, asyncio.Task] = {}

# Local sid lookups — kept in sync with Redis for fast access from hot paths
_local_sid_to_player: dict[str, str] = {}  # sid → player_id
_local_sid_to_room: dict[str, str] = {}    # sid → room_code


# ─── helpers ──────────────────────────────────────────────────────────────────

def _scores(room: dict) -> dict:
    return {p["id"]: p["score"] for p in room["players"]}


async def _enqueue_frame(sid: str, frame_rgb: np.ndarray):
    """Put frame in queue, dropping stale frame if full (stay fresh)."""
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


async def _stop_all_room_workers(room_code: str):
    """Cancel frame workers and timer for every player in this room."""
    sids_in_room = [s for s, c in _local_sid_to_room.items() if c == room_code]
    for sid in sids_in_room:
        await _stop_player_worker(sid)
    timer = _room_timer_tasks.pop(room_code, None)
    if timer:
        timer.cancel()


async def _frame_worker(sid: str, room_code: str):
    """
    Per-player async worker. Pulls frames from the queue, runs CLIP in a
    thread executor, sends confidence feedback, and fires a win when detected.
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

        # Live confidence bar for this player only
        await sio.emit("frame:result", result, to=sid)

        if result["label"] != "none":
            await _handle_round_win(sid, room_code)


# ─── game flow ────────────────────────────────────────────────────────────────

async def _start_next_round(room_code: str):
    room = await redis_state.get_room(room_code)
    if not room or room["phase"] != "game":
        return

    used = set(room["usedObjectIds"])
    available = [oid for oid in get_all_ids() if oid not in used]

    if not available:
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

    if detector:
        detector.set_active_object(object_id, obj)

    timeout = room["settings"]["roundTimeout"]
    await redis_state.set_room(room_code, room)

    timer_task = asyncio.create_task(
        _run_round_timeout(room_code, rnd["id"], timeout)
    )
    _room_timer_tasks[room_code] = timer_task

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
    room = await redis_state.get_room(room_code)
    if not room or room["phase"] != "game":
        return

    current_round = room.get("currentRound")
    if not current_round:
        return

    player_id = _local_sid_to_player.get(sid) or await redis_state.get_sid_player(sid)
    if not player_id:
        return

    # Atomic claim — only the first caller across all workers wins
    won = await redis_state.claim_round_win(room_code, current_round["id"], player_id)
    if not won:
        return

    winner = None
    for p in room["players"]:
        if p["id"] == player_id:
            p["score"] += 1
            winner = p
            break

    if not winner:
        return

    current_round["winnerId"] = player_id
    current_round["winnerName"] = winner["name"]
    room["currentRound"] = current_round
    await redis_state.set_room(room_code, room)

    # Cancel local timer (best-effort; may live on another worker)
    timer = _room_timer_tasks.pop(room_code, None)
    if timer:
        timer.cancel()

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

    points_to_win = room["settings"]["pointsToWin"]
    if winner["score"] >= points_to_win:
        await asyncio.sleep(3)
        await _end_game(room_code, player_id, winner["name"])
        return

    await asyncio.sleep(3)
    await _start_next_round(room_code)


async def _run_round_timeout(room_code: str, round_id: str, timeout_seconds: int):
    await asyncio.sleep(timeout_seconds)

    room = await redis_state.get_room(room_code)
    if not room or room["phase"] != "game":
        return

    current_round = room.get("currentRound")
    if not current_round or current_round["id"] != round_id:
        return

    # Atomic claim using sentinel — prevents both timeout and win firing
    claimed = await redis_state.claim_round_win(room_code, round_id, "__timeout__")
    if not claimed:
        return

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
    room = await redis_state.get_room(room_code)
    if not room:
        return

    room["phase"] = "results"
    await redis_state.set_room(room_code, room)

    await _stop_all_room_workers(room_code)

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

    code = _local_sid_to_room.pop(sid, None) or await redis_state.get_sid_room(sid)
    player_id = _local_sid_to_player.pop(sid, None) or await redis_state.get_sid_player(sid)

    await redis_state.del_sid_room(sid)
    await redis_state.del_sid_player(sid)

    if not code:
        return

    room = await redis_state.get_room(code)
    if not room:
        return

    room["players"] = [p for p in room["players"] if p["id"] != player_id]

    if not room["players"]:
        await redis_state.delete_room(code)
        return

    # Re-assign host if the host disconnected
    if room["hostSid"] == sid:
        new_host = room["players"][0]
        room["hostId"] = new_host["id"]
        new_host_sid = next(
            (s for s, pid in _local_sid_to_player.items() if pid == new_host["id"]),
            None,
        )
        if new_host_sid:
            room["hostSid"] = new_host_sid

    await redis_state.set_room(code, room)
    await sio.emit("player:left", player_id, room=code)


# ─── room management ──────────────────────────────────────────────────────────

@sio.on("room:create")
async def room_create(sid, player_name: str):
    code = generate_room_code()
    while await redis_state.room_exists(code):
        code = generate_room_code()

    player = make_player(player_name)
    room = make_room(code, player, sid)

    await redis_state.set_room(code, room)
    await redis_state.set_sid_room(sid, code)
    await redis_state.set_sid_player(sid, player["id"])
    _local_sid_to_room[sid] = code
    _local_sid_to_player[sid] = player["id"]

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

    room = await redis_state.get_room(code)
    if not room:
        await sio.emit("error", {"message": "Room not found"}, to=sid)
        return

    if room["phase"] != "lobby":
        await sio.emit("error", {"message": "Game already in progress"}, to=sid)
        return

    player = make_player(player_name)
    room["players"].append(player)

    await redis_state.set_room(code, room)
    await redis_state.set_sid_room(sid, code)
    await redis_state.set_sid_player(sid, player["id"])
    _local_sid_to_room[sid] = code
    _local_sid_to_player[sid] = player["id"]

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
    code = _local_sid_to_room.get(sid)
    room = await redis_state.get_room(code) if code else None
    if not room or room["hostSid"] != sid:
        return
    room["players"] = [p for p in room["players"] if p["id"] != player_id]
    await redis_state.set_room(code, room)
    await sio.emit("player:left", player_id, room=code)


@sio.on("settings:update")
async def settings_update(sid, settings: dict):
    code = _local_sid_to_room.get(sid)
    room = await redis_state.get_room(code) if code else None
    if not room or room["hostSid"] != sid:
        return

    allowed_points = [3, 5, 10, 15]
    allowed_timeout = [30, 60, 90, 120]

    if "pointsToWin" in settings and settings["pointsToWin"] in allowed_points:
        room["settings"]["pointsToWin"] = settings["pointsToWin"]
    if "roundTimeout" in settings and settings["roundTimeout"] in allowed_timeout:
        room["settings"]["roundTimeout"] = settings["roundTimeout"]

    await redis_state.set_room(code, room)
    await sio.emit("settings:updated", room["settings"], room=code)


# ─── game flow events ─────────────────────────────────────────────────────────

@sio.on("game:start")
async def game_start(sid):
    code = _local_sid_to_room.get(sid)
    room = await redis_state.get_room(code) if code else None
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
    await redis_state.set_room(code, room)

    # Start a frame worker for every player connected to this process
    for player in room["players"]:
        for s, pid in list(_local_sid_to_player.items()):
            if pid == player["id"]:
                await _start_player_worker(s, code)
                break

    await sio.emit(
        "game:started",
        {"players": room["players"], "settings": room["settings"]},
        room=code,
    )
    print(f"[socket] game:start → {code}")

    await asyncio.sleep(1)
    await _start_next_round(code)


# ─── frame submission ─────────────────────────────────────────────────────────

@sio.on("frame:submit")
async def frame_submit(sid, data: dict):
    # Fast local check — no Redis needed here
    if sid not in player_workers:
        return

    frame_b64: str = data.get("frameData", "")
    if not frame_b64:
        return

    try:
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(frame_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # Resize to CLIP's native input size
        pil_img = pil_img.resize((224, 224), Image.BILINEAR)
        np_img = np.array(pil_img)
    except Exception as e:
        print(f"[frame] decode error for {sid}: {e}")
        return

    await _enqueue_frame(sid, np_img)
