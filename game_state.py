import uuid
import random
import string
from typing import Dict

POINTS_MAP = {1: 100, 2: 60, 3: 30}
POINTS_MIN = 10
CLIP_MIN_SCORE = 0.20

SAMPLE_PROMPTS = [
    "FIND SOMETHING RED",
    "FIND SOMETHING SOFT",
    "FIND SOMETHING WITH WHEELS",
    "FIND SOMETHING YOU WEAR",
    "FIND SOMETHING SHINY",
    "FIND SOMETHING GREEN",
    "FIND SOMETHING ROUND",
    "FIND SOMETHING MADE OF WOOD",
    "FIND SOMETHING THAT OPENS",
    "FIND SOMETHING OLDER THAN YOU",
]


def generate_room_code() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def make_player(name: str) -> dict:
    n = name.strip().upper()
    return {
        "id": str(uuid.uuid4()),
        "name": n,
        "initials": n[0] if n else "?",
        "score": 0,
        "photoUrl": None,
    }


def make_round(prompt: str, round_id: str, time_per_round: int) -> dict:
    return {
        "id": round_id,
        "prompt": prompt,
        "submissions": {},       # player_id -> photoUrl (str or None)
        "submissionOrder": [],   # player_ids in order received (internal)
        "timeRemaining": time_per_round,
    }


def make_room(code: str, host_player: dict, host_sid: str) -> dict:
    return {
        "code": code,
        "players": [host_player],
        "settings": {"rounds": 5, "timePerRound": 60},
        "hostId": host_player["id"],
        "hostSid": host_sid,
        "phase": "lobby",
        "rounds": [],
        "currentRound": 0,
    }


# In-memory state
rooms: Dict[str, dict] = {}         # room_code -> room dict
sid_to_room: Dict[str, str] = {}    # socket_id -> room_code
sid_to_player: Dict[str, str] = {}  # socket_id -> player_id
