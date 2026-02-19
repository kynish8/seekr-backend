import asyncio
import base64
import io
import random

import numpy as np
import socketio
from PIL import Image

from game_state import (
    CLIP_MIN_SCORE,
    POINTS_MAP,
    POINTS_MIN,
    SAMPLE_PROMPTS,
    generate_room_code,
    make_player,
    make_room,
    make_round,
    rooms,
    sid_to_player,
    sid_to_room,
)

# detector is injected by main.py after CLIPDetector is initialized,
# to avoid loading the model twice.
detector = None

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",  # open for local dev; restrict in production
    logger=False,
    engineio_logger=False,
)


# helpers

def _public_round(rnd: dict) -> dict:
    """Strip internal fields before broadcasting a round to clients."""
    return {
        "id": rnd["id"],
        "prompt": rnd["prompt"],
        "submissions": rnd["submissions"],
        "timeRemaining": rnd["timeRemaining"],
    }


def _public_rounds(rounds: list) -> list:
    return [_public_round(r) for r in rounds]


# connect/disconnect

@sio.event
async def connect(sid, environ):
    print(f"[socket] connect {sid}")


@sio.event
async def disconnect(sid):
    print(f"[socket] disconnect {sid}")

    code = sid_to_room.pop(sid, None)
    player_id = sid_to_player.pop(sid, None)

    if not code or code not in rooms:
        return

    room = rooms[code]
    room["players"] = [p for p in room["players"] if p["id"] != player_id]

    if not room["players"]:
        del rooms[code]
        return

    # reassign host if the host left
    if room["hostSid"] == sid:
        new_host = room["players"][0]
        room["hostId"] = new_host["id"]
        # find the new host's socket id
        for s, c in list(sid_to_room.items()):
            if c == code and sid_to_player.get(s) == new_host["id"]:
                room["hostSid"] = s
                break

    await sio.emit("player:left", player_id, room=code)


# room management

@sio.on("room:create")
async def room_create(sid, player_name: str):
    code = generate_room_code()
    # keep generating until unique (should be unlikely to collide)
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

    # tell the joiner their full room state
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

    # tell everyone else a new player arrived
    await sio.emit("player:joined", player, room=code, skip_sid=sid)
    print(f"[socket] room:join → {code} by {player['name']}")


@sio.on("player:remove")
async def player_remove(sid, player_id: str):
    code = sid_to_room.get(sid)
    room = rooms.get(code)
    if not room:
        return
    if room["hostSid"] != sid:
        return  # only host can remove players

    room["players"] = [p for p in room["players"] if p["id"] != player_id]
    await sio.emit("player:left", player_id, room=code)


@sio.on("settings:update")
async def settings_update(sid, settings: dict):
    code = sid_to_room.get(sid)
    room = rooms.get(code)
    if not room or room["hostSid"] != sid:
        return

    allowed_rounds = [3, 5, 8]
    allowed_time = [30, 60, 120]

    if "rounds" in settings and settings["rounds"] in allowed_rounds:
        room["settings"]["rounds"] = settings["rounds"]
    if "timePerRound" in settings and settings["timePerRound"] in allowed_time:
        room["settings"]["timePerRound"] = settings["timePerRound"]

    await sio.emit("settings:updated", room["settings"], room=code)


# game flow

@sio.on("game:start")
async def game_start(sid):
    code = sid_to_room.get(sid)
    room = rooms.get(code)
    if not room:
        return
    if room["hostSid"] != sid:
        return
    if len(room["players"]) < 2:
        await sio.emit(
            "error", {"message": "Need at least 2 players to start"}, to=sid
        )
        return
    if room["phase"] != "lobby":
        return

    n_rounds = room["settings"]["rounds"]
    time_per_round = room["settings"]["timePerRound"]

    prompts = random.sample(SAMPLE_PROMPTS, min(n_rounds, len(SAMPLE_PROMPTS)))
    room["rounds"] = [
        make_round(prompt, str(i + 1), time_per_round)
        for i, prompt in enumerate(prompts)
    ]
    room["currentRound"] = 0
    room["phase"] = "game"

    await sio.emit(
        "game:started",
        {"rounds": _public_rounds(room["rounds"])},
        room=code,
    )

    asyncio.create_task(_run_round_timer(code, 0))
    print(f"[socket] game:start → {code}, {n_rounds} rounds")


async def _run_round_timer(room_code: str, round_index: int):
    room = rooms.get(room_code)
    if not room:
        return

    rnd = room["rounds"][round_index]
    time_per_round = room["settings"]["timePerRound"]

    for remaining in range(time_per_round, -1, -1):
        if room_code not in rooms:
            return  # room was deleted (everyone left)

        rnd["timeRemaining"] = remaining
        await sio.emit("round:update", _public_round(rnd), room=room_code)

        if remaining == 0:
            break
        await asyncio.sleep(1)

    # Advance to next round or end game
    next_index = round_index + 1
    if next_index < len(room["rounds"]):
        room["currentRound"] = next_index
        asyncio.create_task(_run_round_timer(room_code, next_index))
    else:
        room["phase"] = "results"
        await sio.emit(
            "game:ended", {"players": room["players"]}, room=room_code
        )
        print(f"[socket] game:ended → {room_code}")


# photo submission + CLIP scoring

@sio.on("photo:submit")
async def photo_submit(sid, data: dict):
    code = sid_to_room.get(sid)
    room = rooms.get(code)
    player_id = sid_to_player.get(sid)

    if not room or not player_id:
        return
    if room["phase"] != "game":
        return

    round_index = room["currentRound"]
    current_round = room["rounds"][round_index]

    # reject stale submissions for a previous round
    if current_round["id"] != data.get("roundId"):
        return

    # only one submission per player per round
    if player_id in current_round["submissions"]:
        return

    photo_url = data.get("photoUrl", "")
    if not photo_url:
        return

    # decode base64 data URI to numpy RGB array
    try:
        _, b64data = photo_url.split(",", 1)
        img_bytes = base64.b64decode(b64data)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        np_img = np.array(pil_img)
    except Exception as e:
        print(f"[photo] decode error: {e}")
        await sio.emit(
            "submission:rejected", {"reason": "Could not decode image"}, to=sid
        )
        return

    # run CLIP as a validity gate (any detected object passes)
    if detector is not None:
        result = detector.detect_single(np_img)
        if result["label"] == "none":
            await sio.emit(
                "submission:rejected",
                {"reason": "No object detected in photo"},
                to=sid,
            )
            print(f"[photo] rejected for {player_id}: no object")
            return
    # if detector not available (test env), accept everything

    order = current_round["submissionOrder"]
    order.append(player_id)
    position = len(order)
    points = POINTS_MAP.get(position, POINTS_MIN)

    # update player score
    for p in room["players"]:
        if p["id"] == player_id:
            p["score"] += points
            break

    current_round["submissions"][player_id] = photo_url

    print(f"[photo] accepted for {player_id} at position {position} (+{points}pts)")

    await sio.emit("round:update", _public_round(current_round), room=code)

    await sio.emit("players:updated", {"players": room["players"]}, room=code)
