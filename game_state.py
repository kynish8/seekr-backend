import uuid
import random
import string
from typing import Dict


def generate_room_code() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def make_player(name: str) -> dict:
    n = name.strip().upper()
    return {
        "id": str(uuid.uuid4()),
        "name": n,
        "initials": n[0] if n else "?",
        "score": 0,
    }


def make_round(round_id: str, object_id: str, display_name: str) -> dict:
    return {
        "id": round_id,
        "objectId": object_id,
        "displayName": display_name,
        "winnerId": None,
        "winnerName": None,
    }


def make_room(code: str, host_player: dict, host_sid: str) -> dict:
    return {
        "code": code,
        "players": [host_player],
        "settings": {
            "pointsToWin": 5,
            "roundTimeout": 60,   # seconds before round auto-skips
        },
        "hostId": host_player["id"],
        "hostSid": host_sid,
        "phase": "lobby",           # lobby | game | results
        "currentRound": None,       # active round dict
        "roundNumber": 0,
        "usedObjectIds": [],        # prevents repeating objects in a session
        "roundTimerTask": None,     # asyncio.Task handle
    }


# In-memory state
rooms: Dict[str, dict] = {}         # room_code -> room dict
sid_to_room: Dict[str, str] = {}    # socket_id -> room_code
sid_to_player: Dict[str, str] = {}  # socket_id -> player_id
