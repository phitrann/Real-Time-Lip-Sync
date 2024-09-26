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
from typing import List, Dict
import multiprocessing
from loguru import logger
import argparse
import uuid
# Import the worker functions

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc import MediaStreamTrack, VideoStreamTrack
import aiohttp
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.webrtc import HumanPlayer
from app.musereal import MuseReal
# from app.worker import gen_digital_human_video_app, preprocess_digital_human_app

from contextlib import asynccontextmanager
# Ensure that the start method is set before any multiprocessing code

# try:
#     multiprocessing.set_start_method('spawn', force=True)
# except RuntimeError:
#     # Context has already been set; ignore the error if it's already 'spawn'
#     pass


# nerfreals: List[MuseReal] = []
# statreals: List[int] = []

# Initialize MuseReal instances as a dictionary
nerfreals: Dict[str, MuseReal] = {}
pcs = set()
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
    # global nerfreals, statreals
    global nerfreals
    
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
        # nerfreals.append(nerfreal)
        # statreals.append(0)
        
        # Assign a unique sessionid for each MuseReal instance
        sessionid = str(uuid.uuid4())
        nerfreals[sessionid] = nerfreal
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

# # Ensure the 'web' directory exists or adjust the path accordingly
# static_dir = os.path.join(os.path.dirname(__file__), 'static')
# if not os.path.exists(static_dir):
#     os.makedirs(static_dir)

# # Mount static files (if needed)
# app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Set up templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# class DigitalHumanItem(BaseModel):
#     user_id: str  # User identifier
#     request_id: str  # Request ID
#     streamer_id: str  # Digital human ID
#     tts_path: str = ""  # Text path
#     chunk_id: int = 0  # Sentence ID


# class DigitalHumanPreprocessItem(BaseModel):
#     user_id: str  # User identifier
#     request_id: str  # Request ID
#     streamer_id: str  # Digital human ID
#     video_path: str  # Digital human video path

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


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     sessionid = 0  # Default session ID
#     try:
#         while True:
#             data = await websocket.receive_text()
#             message = json.loads(data)
#             if message["type"] == "session":
#                 sessionid = message.get("sessionid", 0)
#             elif message["type"] == "audio":
#                 audio_data_str = message["data"]
#                 audio_data = audio_data_str.encode('latin-1')
#                 nerfreals[sessionid].put_audio_frame(audio_data)
#     except WebSocketDisconnect:
#         print("WebSocket connection closed")

import base64
import cv2
session_video_queues: Dict[int, asyncio.Queue] = {}  # To store video queues per sessionid
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sessionid = None  # To store the sessionid
    video_queue = None
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            msg_type = data.get("type")
            if msg_type == "session":
                sessionid = data.get("sessionid")
                if sessionid is not None:
                    if sessionid not in nerfreals:
                        # Initialize a new MuseReal instance for this session
                        nerfreal = MuseReal(app.state.opt)
                        nerfreals[sessionid] = nerfreal
                        print(f"Session {sessionid} connected.")
                        # Start rendering for the new session
                        nerfreals[sessionid].start_rendering()
                        # Optionally, send confirmation back to the client
                        await websocket.send_json({"type": "session_ack", "sessionid": sessionid})
                    else:
                        print(f"Session {sessionid} already exists.")
                        await websocket.send_json({"type": "session_ack", "sessionid": sessionid})
                else:
                    await websocket.send_json({"type": "error", "message": "No sessionid provided."})
                    
            elif msg_type == "audio":
                if sessionid is None:
                    await websocket.send_json({"type": "error", "message": "Sessionid not set."})
                    continue
                audio_data_str = data.get("data")
                if audio_data_str:
                    try:
                        # Decode audio data from base64
                        audio_data = base64.b64decode(audio_data_str)
                    except Exception as e:
                        await websocket.send_json({"type": "error", "message": "Invalid audio data encoding."})
                        continue
                    # Pass audio data to MuseReal instance
                    nerfreals[sessionid].put_audio_frame(audio_data)
                    # Retrieve video frame from MuseReal
                    video_frame = await nerfreals[sessionid].get_video_frame()
                    if video_frame is not None:
                        # Encode the video frame as JPEG
                        _, img_encoded = cv2.imencode('.jpg', video_frame)
                        img_bytes = img_encoded.tobytes()
                        # Encode to base64 for JSON transmission
                        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                        # Send the video frame back to the client
                        await websocket.send_json({
                            'type': 'video',
                            'data': img_base64
                        })
            elif msg_type == "end":
                if sessionid is not None:
                    logger.info(f"Session {sessionid} ended.")
                    # Clean up session
                    if sessionid in nerfreals:
                        del nerfreals[sessionid]
            else:
                await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})
    except WebSocketDisconnect:
        if sessionid is not None:
            logger.info(f"Session {sessionid} disconnected.")
            if sessionid in nerfreals:
                del nerfreals[sessionid]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

async def send_video_frames(websocket: WebSocket, sessionid: int, video_queue: asyncio.Queue):
    """
    Sends video frames from the video_queue to the client via WebSocket.
    """
    try:
        while True:
            img_bytes = await video_queue.get()
            # Encode the image as base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            # Send the image back to the client
            await websocket.send_json({
                'type': 'video',
                'data': img_base64
            })
    except Exception as e:
        logger.error(f"Error sending video frames: {e}")

@app.post("/offer")
async def offer_endpoint(request: Request):
    global nerfreals, pcs
    
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Generate a unique sessionid
    sessionid = str(uuid.uuid4())
    
    # Create a new MuseReal instance
    nerfreal = MuseReal(app.state.opt)
    nerfreals[sessionid] = nerfreal

    # Create a new RTCPeerConnection
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)
            if sessionid in nerfreals:
                del nerfreals[sessionid]
    
    # Add transceivers
    pc.addTransceiver('video', direction='sendonly')
    pc.addTransceiver('audio', direction='sendonly')

    # Add tracks
    player = HumanPlayer(nerfreals[sessionid])
    pc.addTrack(player.video)
    pc.addTrack(player.audio)

    # Set codec preferences for video
    for transceiver in pc.getTransceivers():
        if transceiver.kind == 'video':
            capabilities = RTCRtpSender.getCapabilities('video')
            preferences = [codec for codec in capabilities.codecs if codec.name in ('H264', 'VP8')]
            transceiver.setCodecPreferences(preferences)
            transceiver.direction = 'sendonly'
        elif transceiver.kind == 'audio':
            transceiver.direction = 'sendonly'

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(content={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid})

# @app.post("/offer")
# async def offer(request: Request):
#     global nerfreals, statreals, pcs
    
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

#     # Add transceivers
#     pc.addTransceiver('video', direction='sendonly')
#     pc.addTransceiver('audio', direction='sendonly')

#     # Add tracks
#     player = HumanPlayer(nerfreals[sessionid])
#     pc.addTrack(player.video)
#     pc.addTrack(player.audio)

#     # Set codec preferences for video
#     for transceiver in pc.getTransceivers():
#         if transceiver.kind == 'video':
#             capabilities = RTCRtpSender.getCapabilities('video')
#             preferences = [codec for codec in capabilities.codecs if codec.name in ('H264', 'VP8')]
#             transceiver.setCodecPreferences(preferences)
#             transceiver.direction = 'sendonly'
#         elif transceiver.kind == 'audio':
#             transceiver.direction = 'sendonly'

#     await pc.setRemoteDescription(offer)
#     answer = await pc.createAnswer()
#     await pc.setLocalDescription(answer)

#     return JSONResponse(content={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid})


@app.post("/humanaudio")
async def humanaudio(
    file: UploadFile = File(...), 
    sessionid: int = Form(0),
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
