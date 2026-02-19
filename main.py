import fastapi
import asyncio
import json
import socketio

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import MediaStreamError
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

from clip_detector import CLIPDetector
from socket_server import sio
import socket_server

# Initialize CLIP detector (shared with socket_server to avoid loading twice)
detector = CLIPDetector()
socket_server.detector = detector

app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs = set()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/offer")
async def offer(request: Request):
    data = await request.json()

    offer = RTCSessionDescription(
        sdp=data["sdp"],
        type=data["type"]
    )

    pc = RTCPeerConnection(
        configuration=RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302"])
            ]
        )
    )

    pcs.add(pc)

    relay = MediaRelay()
    data_channel = pc.createDataChannel("results")

    @data_channel.on("open")
    def on_open():
        print("✅ DataChannel is open")

    @data_channel.on("close")
    def on_close():
        print("❌ DataChannel closed")

    @pc.on("track")
    async def on_track(track):
        if track.kind != "video":
            return

        track = relay.subscribe(track)

        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                break

            img = frame.to_ndarray(format="rgb24")
            result = detector.detect(img)
            print(result)

            if data_channel.readyState == "open":
                data_channel.send(json.dumps(result))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    }


# wrap FastAPI with Socket.io so all non-socket HTTP requests pass through to app
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:socket_app", host="0.0.0.0", port=3001)
