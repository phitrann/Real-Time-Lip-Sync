# from fastapi import FastAPI
# from fastapi.exceptions import RequestValidationError
# from fastapi.responses import PlainTextResponse
# from loguru import logger
# from pydantic import BaseModel


# from app.worker import gen_digital_human_video_app, preprocess_digital_human_app


# app = FastAPI()


# class DigitalHumanItem(BaseModel):
#     user_id: str  # User 识别号，用于区分不用的用户调用
#     request_id: str  # 请求 ID，用于生成 TTS & 数字人
#     streamer_id: str  # 数字人 ID
#     tts_path: str = ""  # 文本
#     chunk_id: int = 0  # 句子 ID


# class DigitalHumanPreprocessItem(BaseModel):
#     user_id: str  # User 识别号，用于区分不用的用户调用
#     request_id: str  # 请求 ID，用于生成 TTS & 数字人
#     streamer_id: str  # 数字人 ID
#     video_path: str  # 数字人视频


# @app.post("/digital_human/gen")
# async def get_digital_human(dg_item: DigitalHumanItem):
#     """Generate digital human video"""
#     save_tag = (
#         dg_item.request_id + ".mp4" if dg_item.chunk_id == 0 else dg_item.request_id + f"-{str(dg_item.chunk_id).zfill(8)}.mp4"
#     )
#     mp4_path = await gen_digital_human_video_app(dg_item.streamer_id, dg_item.tts_path, save_tag)
#     logger.info(f"digital human mp4 path = {mp4_path}")
#     return {"user_id": dg_item.user_id, "request_id": dg_item.request_id, "digital_human_mp4_path": mp4_path}


# @app.post("/digital_human/preprocess")
# async def preprocess_digital_human(preprocess_item: DigitalHumanPreprocessItem):
#     """Digital human video preprocessing for adding new digital humans"""

#     _ = await preprocess_digital_human_app(str(preprocess_item.streamer_id), preprocess_item.video_path)

#     logger.info(f"digital human process for {preprocess_item.streamer_id} done")
#     return {"user_id": preprocess_item.user_id, "request_id": preprocess_item.request_id}

# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request, exc):
#     """调 API 入参错误的回调接口

#     Args:
#         request (_type_): _description_
#         exc (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     logger.info(request)
#     logger.info(exc)
#     return PlainTextResponse(str(exc), status_code=400)


# @app.get("/digital_human/check")
# async def check_server():
#     return {"message": "server enabled"}

# app.py

from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, Request, HTTPException, File, UploadFile, Form
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import asyncio
import json
import os
import time
import numpy as np

from threading import Thread, Event
import multiprocessing
from loguru import logger
import argparse
# Import the worker functions
from typing import List

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc import MediaStreamTrack
import aiohttp

from app.webrtc import HumanPlayer
from app.musereal import MuseReal

from contextlib import asynccontextmanager
# Ensure that the start method is set before any multiprocessing code

# try:
#     multiprocessing.set_start_method('spawn', force=True)
# except RuntimeError:
#     # Context has already been set; ignore the error if it's already 'spawn'
#     pass


nerfreals: List[MuseReal] = []
statreals: List[int] = []
pcs = set()

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--avatar_id', type=str, default='avator_1')
    parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--customvideo_config', type=str, default='')
    parser.add_argument('--transport', type=str, default='webrtc')
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream')
    parser.add_argument('--max_session', type=int, default=1)
    parser.add_argument('--listenport', type=int, default=8010)
    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)
    
    opt, _ = parser.parse_known_args()
    return opt
    

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks: Initialize MuseReal instances
    opt = get_args()
    # Load custom video configuration if provided
    opt.customopt = []
    if opt.customvideo_config != '':
        with open(opt.customvideo_config, 'r') as file:
            opt.customopt = json.load(file)

    # Add opt to FastAPI app state so it can be accessed inside the lifespan
    app.state.opt = opt
    
    for _ in range(opt.max_session):
        nerfreal = MuseReal(opt)
        nerfreals.append(nerfreal)
        statreals.append(0)
    pagename = 'webrtcapi.html' if opt.transport == 'webrtc' else 'echoapi.html'
    print(f'start http server; http://<serverip>:{opt.listenport}/{pagename}')


    # Yield to let the app run
    yield

    # Shutdown tasks: Close WebRTC connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    
app = FastAPI(lifespan=lifespan)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the 'web' directory exists or adjust the path accordingly
web_dir = os.path.join(os.path.dirname(__file__), 'web')
if not os.path.exists(web_dir):
    os.makedirs(web_dir)

# Mount static files (if needed)
# app.mount("/", StaticFiles(directory=web_dir), name="static")

def get_nerfreals():
    return nerfreals

class DigitalHumanItem(BaseModel):
    user_id: str  # User identifier
    request_id: str  # Request ID
    streamer_id: str  # Digital human ID
    tts_path: str = ""  # Text path
    chunk_id: int = 0  # Sentence ID


class DigitalHumanPreprocessItem(BaseModel):
    user_id: str  # User identifier
    request_id: str  # Request ID
    streamer_id: str  # Digital human ID
    video_path: str  # Digital human video path


@app.websocket("/humanecho")
async def echo_socket(websocket: WebSocket):
    await websocket.accept()
    print('WebSocket connected to /humanecho!')
    while True:
        message = await websocket.receive_text()
        if not message:
            continue
        # Pass the message to your processing function
        nerfreals[0].put_msg_txt(message)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message["type"] == "audio":
                sessionid = message.get("sessionid", 0)
                audio_data = message["data"]
                nerfreals[sessionid].put_audio_frame(audio_data)
            elif message["type"] == "video":
                # Handle incoming video frame
                pass
    except WebSocketDisconnect:
        print("WebSocket connection closed")

class OfferModel(BaseModel):
    sdp: str
    type: str
    
class RelayTrack(MediaStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track
        self.relay = MediaRelay()

    async def recv(self):
        frame = await self.track.recv()
        return self.relay.relay(frame)

@app.post("/offer")
async def offer(params: OfferModel):
    offer = RTCSessionDescription(sdp=params.sdp, type=params.type)

    sessionid = len(nerfreals)
    for index, value in enumerate(statreals):
        if value == 0:
            sessionid = index
            break
    if sessionid >= len(nerfreals):
        print('Reached max session limit')
        return JSONResponse(content={"error": "Max sessions reached"}, status_code=400)
    statreals[sessionid] = 1

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.pop(str(sessionid), None)
            statreals[sessionid] = 0

    # Add handlers for incoming media (audio)
    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            print("Received audio track")
            @track.on("ended")
            async def on_ended():
                print("Audio track ended")
        elif track.kind == "video":
            print("Received video track")
            pc.addTrack(RelayTrack(track))

   # Create a video stream from MuseReal
    player = HumanPlayer(nerfreals[sessionid])
    pc.addTrack(player.video)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(content={
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "sessionid": sessionid
    })

# @app.post("/offer")
# async def offer(request: Request):
#     params = await request.json()
#     offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

#     sessionid = len(nerfreals)
#     for index, value in enumerate(statreals):
#         if value == 0:
#             sessionid = index
#             break

#     if sessionid >= len(nerfreals):
#         return JSONResponse(status_code=400, content={"message": "Max sessions reached"})
    
#     statreals[sessionid] = 1

#     pc = RTCPeerConnection()
#     pcs.add(pc)

#     @pc.on("connectionstatechange")
#     async def on_connectionstatechange():
#         print(f"Connection state is {pc.connectionState}")
#         if pc.connectionState in ["failed", "closed"]:
#             await pc.close()
#             pcs.discard(pc)
#             statreals[sessionid] = 0

#     player = HumanPlayer(nerfreals[sessionid])
#     audio_sender = pc.addTrack(player.audio)
#     video_sender = pc.addTrack(player.video)

#     capabilities = RTCRtpSender.getCapabilities("video")
#     preferences = list(filter(lambda x: x.name in ["H264", "VP8", "rtx"], capabilities.codecs))
#     transceiver = pc.getTransceivers()[1]
#     transceiver.setCodecPreferences(preferences)

#     await pc.setRemoteDescription(offer)
#     answer = await pc.createAnswer()
#     await pc.setLocalDescription(answer)

#     return JSONResponse(content={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid})


@app.post("/humanaudio")
async def humanaudio(
    file: UploadFile = File(...), 
    sessionid: int = Form(0),
    # nerfreals: List[MuseReal] = Depends(get_nerfreals)
):
    global nerfreals
    try:
        # Check if sessionid is valid
        if sessionid < 0 or sessionid >= len(nerfreals):
            return JSONResponse(content={"session_id": sessionid, "code": -1, "msg": "Invalid session ID"})
        
        filebytes = await file.read()
        
        nerfreals[sessionid].put_audio_file(filebytes)

        return JSONResponse(content={"code": 0, "msg": "ok"})
    except Exception as e:
        return JSONResponse(content={"code": -1, "msg": "err", "data": str(e)})


@app.post("/set_audiotype")
async def set_audiotype(request: Request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    if params['type'] == 'start_record':
        nerfreals[sessionid].start_recording("data/record_lasted.mp4")
    elif params['type'] == 'end_record':
        nerfreals[sessionid].stop_recording()
        
    return JSONResponse(content={"code": 0, "data": "ok"})


@app.post("/record")
async def record(request: Request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    if params['type'] == 'start_record':
        nerfreals[sessionid].start_recording("data/record_lasted.mp4")
    elif params['type'] == 'end_record':
        nerfreals[sessionid].stop_recording()
    return JSONResponse(content={"code": 0, "data": "ok"})


@app.post("/is_speaking")
async def is_speaking(request: Request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    is_speaking = nerfreals[sessionid].is_speaking()

    return JSONResponse(content={"code": 0, "data": is_speaking})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Callback interface for API input parameter errors"""
    logger.info(request)
    logger.info(exc)
    return PlainTextResponse(str(exc), status_code=400)


@app.get("/digital_human/check")
async def check_server():
    return {"message": "server enabled"}

async def post(url, data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        print(f'Error: {e}')

async def run(push_url):
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreals[0])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url, {"sdp": pc.localDescription.sdp})
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type='answer'))
