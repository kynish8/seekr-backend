import os

import fastapi
import socketio
from fastapi.middleware.cors import CORSMiddleware

import redis_state
from clip_detector import CLIPDetector
from socket_server import sio
import socket_server

app = fastapi.FastAPI()

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await redis_state.init()
    # Load CLIP inside the worker process (after AsyncRedisManager forks)
    detector = CLIPDetector()
    socket_server.detector = detector


@app.get("/")
def read_root():
    return {"status": "ok"}


# Wrap FastAPI with Socket.io — all non-socket HTTP requests pass through
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:socket_app", host="0.0.0.0", port=3001, reload=False)
