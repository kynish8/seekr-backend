"""
Redis-backed state for Seekr.

Replaces the in-memory rooms / sid_to_* dicts from game_state.py.
All room data is stored as JSON with a 2-hour TTL so crashed processes
don't leave stale rooms, and any uvicorn worker can read shared state.
"""

import json
import os
from typing import Optional

import redis.asyncio as aioredis

_client: aioredis.Redis | None = None

ROOM_TTL = 7200  # seconds (2 hours)
_SKIP = {"roundTimerTask"}  # asyncio.Task — not serialisable


async def init(url: str | None = None) -> None:
    global _client
    _client = aioredis.from_url(
        url or os.getenv("REDIS_URL", "redis://localhost:6379"),
        decode_responses=False,
    )
    # Verify connection
    await _client.ping()
    print("[redis] connected")


def _r() -> aioredis.Redis:
    if _client is None:
        raise RuntimeError("Redis not initialised — call await redis_state.init() first")
    return _client


# ── Room CRUD ─────────────────────────────────────────────────────────────────

async def get_room(code: str) -> Optional[dict]:
    raw = await _r().get(f"room:{code}")
    return json.loads(raw) if raw else None


async def set_room(code: str, room: dict) -> None:
    data = {k: v for k, v in room.items() if k not in _SKIP}
    await _r().set(f"room:{code}", json.dumps(data), ex=ROOM_TTL)


async def delete_room(code: str) -> None:
    await _r().delete(f"room:{code}")


async def room_exists(code: str) -> bool:
    return bool(await _r().exists(f"room:{code}"))


# ── Atomic round winner claim ─────────────────────────────────────────────────

async def claim_round_win(code: str, round_id: str, claimer_id: str) -> bool:
    """
    Atomically record the first winner (or timeout) for a round.
    Returns True only for the very first caller — safe across multiple workers.
    Uses SET NX so concurrent CLIP detections can't both fire a win event.
    """
    key = f"round_winner:{code}:{round_id}"
    result = await _r().set(key, claimer_id, nx=True, ex=300)
    return result is True


# ── sid → room / player mappings ─────────────────────────────────────────────

async def set_sid_room(sid: str, code: str) -> None:
    await _r().set(f"sid_room:{sid}", code, ex=ROOM_TTL)


async def get_sid_room(sid: str) -> Optional[str]:
    v = await _r().get(f"sid_room:{sid}")
    return v.decode() if v else None


async def del_sid_room(sid: str) -> None:
    await _r().delete(f"sid_room:{sid}")


async def set_sid_player(sid: str, player_id: str) -> None:
    await _r().set(f"sid_player:{sid}", player_id, ex=ROOM_TTL)


async def get_sid_player(sid: str) -> Optional[str]:
    v = await _r().get(f"sid_player:{sid}")
    return v.decode() if v else None


async def del_sid_player(sid: str) -> None:
    await _r().delete(f"sid_player:{sid}")
