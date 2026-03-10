"""
Metrics tracking for Seekr, backed by Redis.
All functions are best-effort — they never raise on Redis errors.
"""

import time
import uuid
from typing import Optional

import redis_state


# ── Key helpers ───────────────────────────────────────────────────────────────

def _counter_key(name: str) -> str:
    return f"metrics:counter:{name}"


def _gauge_key(name: str) -> str:
    return f"metrics:gauge:{name}"


def _hist_key(name: str) -> str:
    return f"metrics:hist:{name}"


def _ts_key(name: str, room_code: str) -> str:
    return f"metrics:ts:{name}:{room_code}"


def _unique_member() -> str:
    return f"{time.time():.6f}:{uuid.uuid4().hex[:8]}"


# ── Write operations ─────────────────────────────────────────────────────────

async def incr(name: str, amount: int = 1) -> None:
    try:
        await redis_state._r().incrby(_counter_key(name), amount)
    except Exception:
        pass


async def gauge_incr(name: str, amount: int = 1) -> None:
    try:
        await redis_state._r().incrby(_gauge_key(name), amount)
    except Exception:
        pass


async def gauge_decr(name: str, amount: int = 1) -> None:
    try:
        await redis_state._r().decrby(_gauge_key(name), amount)
    except Exception:
        pass


async def hist_observe(name: str, value: float) -> None:
    try:
        await redis_state._r().zadd(_hist_key(name), {_unique_member(): value})
    except Exception:
        pass


async def object_incr(metric: str, object_id: str, amount: int = 1) -> None:
    try:
        await redis_state._r().hincrby(f"metrics:object:{metric}", object_id, amount)
    except Exception:
        pass


async def set_timestamp(name: str, room_code: str) -> None:
    try:
        await redis_state._r().set(_ts_key(name, room_code), str(time.time()), ex=7200)
    except Exception:
        pass


async def get_timestamp(name: str, room_code: str) -> Optional[float]:
    try:
        v = await redis_state._r().get(_ts_key(name, room_code))
        return float(v) if v else None
    except Exception:
        return None


async def del_timestamp(name: str, room_code: str) -> None:
    try:
        await redis_state._r().delete(_ts_key(name, room_code))
    except Exception:
        pass


# ── Read operations (for /stats endpoint) ────────────────────────────────────

_COUNTER_NAMES = [
    "lobby_created", "lobby_abandoned",
    "game_started", "game_completed", "game_abandoned",
    "round_started", "round_won", "round_timed_out",
    "frames_processed",
]

_GAUGE_NAMES = ["connections_active", "rooms_active", "players_in_game"]

_HIST_NAMES = [
    "lobby_player_count", "lobby_wait_seconds",
    "game_duration_seconds", "game_rounds_played",
    "round_win_time_seconds", "frame_confidence",
]


async def get_all_stats() -> dict:
    r = redis_state._r()

    # Counters
    pipe = r.pipeline()
    for name in _COUNTER_NAMES:
        pipe.get(_counter_key(name))
    values = await pipe.execute()
    counters = {
        name: int(val) if val else 0
        for name, val in zip(_COUNTER_NAMES, values)
    }

    # Gauges
    pipe = r.pipeline()
    for name in _GAUGE_NAMES:
        pipe.get(_gauge_key(name))
    values = await pipe.execute()
    gauges = {
        name: int(val) if val else 0
        for name, val in zip(_GAUGE_NAMES, values)
    }

    # Histograms
    histograms = {}
    for name in _HIST_NAMES:
        histograms[name] = await _summarize_sorted_set(r, _hist_key(name))

    # Per-object counters
    object_counters = {}
    for metric in ("selected", "found", "timed_out"):
        raw = await r.hgetall(f"metrics:object:{metric}")
        object_counters[metric] = {
            (k.decode() if isinstance(k, bytes) else k): int(v)
            for k, v in raw.items()
        }

    # Per-object find-time histograms
    object_find_times = {}
    for oid in object_counters.get("selected", {}):
        summary = await _summarize_sorted_set(r, _hist_key(f"object_find_time:{oid}"))
        if summary["count"] > 0:
            object_find_times[oid] = summary

    return {
        "counters": counters,
        "gauges": gauges,
        "histograms": histograms,
        "objects": {
            "counters": object_counters,
            "find_time_seconds": object_find_times,
        },
    }


async def _summarize_sorted_set(r, key: str) -> dict:
    count = await r.zcard(key)
    if count == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p95": None, "p99": None}

    members = await r.zrangebyscore(key, "-inf", "+inf", withscores=True)
    scores = [score for _, score in members]
    total = sum(scores)

    return {
        "count": count,
        "min": round(scores[0], 3),
        "max": round(scores[-1], 3),
        "mean": round(total / count, 3),
        "p50": round(_percentile(scores, 50), 3),
        "p95": round(_percentile(scores, 95), 3),
        "p99": round(_percentile(scores, 99), 3),
    }


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * p / 100)))
    return sorted_values[k]
