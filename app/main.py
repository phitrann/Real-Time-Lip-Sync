# # app/main.py
# from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, WebSocket
# from fastapi.responses import StreamingResponse
# import io
# import numpy as np
# import cv2
# import torch
# from pathlib import Path
# import sys
# import asyncio
# import os
# import pickle
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm

# # Add the project root to the Python path
# project_root = Path(__file__).resolve().parents[1]
# sys.path.append(str(project_root))

# from rtlipsync.inference.realtime_inference import Avatar, load_all_model
# from rtlipsync.utils.utils import get_file_type, get_video_fps
# from rtlipsync.utils.utils import datagen   

# app = FastAPI()

# # Load models
# audio_processor, vae, unet, pe = load_all_model()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# timesteps = torch.tensor([0], device=device)

# # Move models to GPU and use half precision
# pe = pe.half()
# vae.vae = vae.vae.half()
# unet.model = unet.model.half()

# # Create a thread pool for background processing
# thread_pool = ThreadPoolExecutor(max_workers=4)

# class AsyncAvatar(Avatar):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.output_queue = asyncio.Queue()

#     async def prepare_video(self):
#         # Call the original prepare_material method in a separate thread
#         loop = asyncio.get_running_loop()
#         await loop.run_in_executor(None, self.prepare_material)

#     @torch.no_grad()
#     async def inference(self, audio_path, fps, skip_save_images=True):
#         # Extract audio features
#         whisper_feature = await asyncio.to_thread(audio_processor.audio2feat, audio_path)
#         whisper_chunks = await asyncio.to_thread(audio_processor.feature2chunks, whisper_feature, fps)

#         # Prepare generator
#         gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
#         video_num = len(whisper_chunks)

#         for i, (whisper_batch, latent_batch) in enumerate(gen):
#             # Convert to tensors
#             audio_feature_batch = torch.from_numpy(whisper_batch).to(device=unet.device, dtype=unet.model.dtype)
#             audio_feature_batch = pe(audio_feature_batch)
#             latent_batch = latent_batch.to(device=unet.device, dtype=unet.model.dtype)

#             pred_latents = unet.model(
#                 latent_batch,
#                 torch.tensor([0], device=unet.device),
#                 encoder_hidden_states=audio_feature_batch
#             ).sample

#             recon = vae.decode_latents(pred_latents)

#             # Process and enqueue frames
#             for j, res_frame in enumerate(recon):
#                 # res_frame = res_frame.cpu().numpy()
#                 await self.output_queue.put(res_frame)


#     async def get_frame(self):
#         return await self.output_queue.get()

# @app.post("/lip_sync")
# async def lip_sync(
#     avatar_id: str = Form(...),
#     video: UploadFile = File(...),
#     audio: UploadFile = File(...),
#     fps: int = Form(25),
#     batch_size: int = Form(4),
#     bbox_shift: int = Form(5),
# ):
#     # Save uploaded files temporarily
#     video_path = f"temp_{avatar_id}_video.mp4"
#     audio_path = f"temp_{avatar_id}_audio.wav"
    
#     with open(video_path, "wb") as buffer:
#         buffer.write(await video.read())
#     with open(audio_path, "wb") as buffer:
#         buffer.write(await audio.read())

#     # Create AsyncAvatar instance
#     avatar = AsyncAvatar(
#         avatar_id=avatar_id,
#         video_path=video_path,
#         bbox_shift=5,
#         batch_size=batch_size,
#         preparation=True
#     )

#     # Prepare video and start processing in the background
#     # await avatar.prepare_video()
#     avatar.prepare_material()
#     background_tasks = BackgroundTasks()
#     background_tasks.add_task(avatar.inference, audio_path, fps)

#     # Clean up temporary files
#     background_tasks.add_task(Path(video_path).unlink, missing_ok=True)
#     background_tasks.add_task(Path(audio_path).unlink, missing_ok=True)

#     # Stream the output frames
#     async def generate_frames():
#         while True:
#             try:
#                 frame = await asyncio.wait_for(avatar.get_frame(), timeout=1.0)
#                 _, buffer = cv2.imencode('.jpg', frame)
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             except asyncio.TimeoutError:
#                 break
#             except Exception as e:
#                 print(f"Error generating frames: {e}")
#                 break

#     return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame", background=background_tasks)

# @app.get("/")
# async def root():
#     return {"message": "Welcome to RTLipSync API"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=1200, workers=4)


from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from loguru import logger
from pydantic import BaseModel


from .worker import gen_digital_human_video_app, preprocess_digital_human_app


app = FastAPI()


class DigitalHumanItem(BaseModel):
    user_id: str  # User 识别号，用于区分不用的用户调用
    request_id: str  # 请求 ID，用于生成 TTS & 数字人
    streamer_id: str  # 数字人 ID
    tts_path: str = ""  # 文本
    chunk_id: int = 0  # 句子 ID


class DigitalHumanPreprocessItem(BaseModel):
    user_id: str  # User 识别号，用于区分不用的用户调用
    request_id: str  # 请求 ID，用于生成 TTS & 数字人
    streamer_id: str  # 数字人 ID
    video_path: str  # 数字人视频


@app.post("/digital_human/gen")
async def get_digital_human(dg_item: DigitalHumanItem):
    """生成数字人视频"""
    save_tag = (
        dg_item.request_id + ".mp4" if dg_item.chunk_id == 0 else dg_item.request_id + f"-{str(dg_item.chunk_id).zfill(8)}.mp4"
    )
    mp4_path = await gen_digital_human_video_app(dg_item.streamer_id, dg_item.tts_path, save_tag)
    logger.info(f"digital human mp4 path = {mp4_path}")
    return {"user_id": dg_item.user_id, "request_id": dg_item.request_id, "digital_human_mp4_path": mp4_path}


@app.post("/digital_human/preprocess")
async def preprocess_digital_human(preprocess_item: DigitalHumanPreprocessItem):
    """数字人视频预处理，用于新增数字人"""

    _ = await preprocess_digital_human_app(str(preprocess_item.streamer_id), preprocess_item.video_path)

    logger.info(f"digital human process for {preprocess_item.streamer_id} done")
    return {"user_id": preprocess_item.user_id, "request_id": preprocess_item.request_id}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """调 API 入参错误的回调接口

    Args:
        request (_type_): _description_
        exc (_type_): _description_

    Returns:
        _type_: _description_
    """
    logger.info(request)
    logger.info(exc)
    return PlainTextResponse(str(exc), status_code=400)


@app.get("/digital_human/check")
async def check_server():
    return {"message": "server enabled"}