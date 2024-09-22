from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
import io
import numpy as np
import cv2
import torch
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from rtlipsync.inference.realtime_inference import Avatar, load_all_model
from rtlipsync.utils.utils import get_file_type, get_video_fps

app = FastAPI()

# Load models
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.post("/lip_sync")
async def lip_sync(
    avatar_id: str = Form(...),
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    fps: int = Form(25),
    batch_size: int = Form(4)
):
    # Save uploaded files temporarily
    video_path = f"temp_{avatar_id}_video.mp4"
    audio_path = f"temp_{avatar_id}_audio.wav"
    
    with open(video_path, "wb") as buffer:
        buffer.write(await video.read())
    with open(audio_path, "wb") as buffer:
        buffer.write(await audio.read())

    # Create Avatar instance
    avatar = Avatar(
        avatar_id=avatar_id,
        video_path=video_path,
        bbox_shift=(0, 0, 0, 0),  # You might want to make this configurable
        batch_size=batch_size,
        preparation=True
    )

    # Process lip sync
    output_queue = avatar.inference(
        audio_path=audio_path,
        out_vid_name=None,
        fps=fps,
        skip_save_images=True
    )

    # Clean up temporary files
    Path(video_path).unlink()
    Path(audio_path).unlink()

    # Stream the output frames
    def generate_frames():
        while True:
            try:
                frame = output_queue.get(timeout=1)
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except:
                break

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def root():
    return {"message": "Welcome to RTLipSync API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1200)