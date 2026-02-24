import uuid
import random
import string


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
            "roundTimeout": 60,
        },
        "hostId": host_player["id"],
        "hostSid": host_sid,
        "phase": "lobby",
        "currentRound": None,
        "roundNumber": 0,
        "usedObjectIds": [],
    }
