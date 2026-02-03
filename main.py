import fastapi
import asyncio
import time
import torch
import clip
from PIL import Image
from collections import deque
import cv2
from fastapi.responses import StreamingResponse
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from fastapi import Request
from aiortc.contrib.media import MediaRelay
import json
from clip_detector import CLIPDetector,WINDOW_SIZE, OBJECTS, GLOBAL_NULLS 
from fastapi.middleware.cors import CORSMiddleware

from aiortc.mediastreams import MediaStreamError
from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer

detector = CLIPDetector()
app = fastapi.FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # or ["*"] for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

pcs = set()

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
            frame = await track.recv()
            try:
                frame = await track.recv()
            except MediaStreamError:
                break

            img = frame.to_ndarray(format="rgb24")
            result = detector.detect(img)
            print(result)

            try:
                data_channel.send(json.dumps(result))
            except Exception:
                pass
            if data_channel.readyState == "open":
                data_channel.send(json.dumps(result))


    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
