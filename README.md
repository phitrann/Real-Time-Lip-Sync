# Real-Time-Lip-Sync

This project makes use of the MuseTalk model to perform real-time lip-syncing on a video and provides an API for generating digital human videos.

## Project Structure
Below is a detailed explanation of the project structure:
```bash
├── app
│   ├── __pycache__              # Python cache files
│   ├── config.py                # Configuration file for the project
│   ├── museasr.py               # Audio processing module for ASR functionality
│   ├── musereal.py              # Main class handling real-time lip-sync inference using MuseTalk model
│   ├── realtime_inference.py    # Script to run inference in real-time
│   ├── webrtc.py                # WebRTC module to handle real-time video and audio streaming
│   └── worker.py                # Worker thread for handling background processes
├── assets
│   └── musetalk_arc.jpg         # Architecture diagram for MuseTalk (Optional)
├── configs
│   └── inference                # Configuration files for inference settings
├── data
│   ├── audio                    # Folder to store input audio files
│   └── video                    # Folder to store input video files
├── models
│   ├── drivers                  # Pre-trained models and drivers for the inference process
│   ├── dwpose                   # Pose estimation models for aligning facial landmarks
│   ├── face-parse-bisent        # Face parsing models for detecting regions of the face
│   ├── musetalk                 # MuseTalk model files for lip-sync functionality
│   ├── sd-vae-ft-mse            # Variational autoencoder models for fine-tuning
│   ├── whisper                  # Whisper ASR model used for speech recognition
│   └── README.md                # Readme file for models directory
├── results
│   ├── avatars                  # Generated avatars after running the inference
│   ├── sun                      # Processed results for the "sun" video
│   ├── yongen                   # Processed results for the "yongen" video
│   ├── sun_sun.mp4              # Final output video for "sun"
│   └── yongen_yongen.mp4        # Final output video for "yongen"
├── rtlipsync
│   ├── inference                # Inference-related scripts and modules for real-time lip-sync
│   ├── models                   # Additional models used for lip-sync
│   ├── utils                    # Utility functions for the project
│   └── whisper                  # Whisper model integration for ASR task
├── scripts
│   ├── download_models.py       # Script to download necessary pre-trained models
│   └── __init__.py              # Init file for the `scripts` module
├── web
│   ├── client.js                # JavaScript client for WebRTC integration
│   ├── webrtcapi.html           # HTML interface for interacting with the WebRTC API
│   └── webrtc.html              # HTML interface for WebRTC video streaming
├── cookbook.ipynb               # Jupyter notebook with experimental code snippets
├── deploy.sh                    # Shell script for deployment setup
├── Dockerfile                   # Dockerfile to containerize the project
├── main.py                      # Main entry point of the application
├── output_video.mp4             # Output video generated after running the model
├── pyproject.toml               # Project configuration file for Python dependencies
├── README.md                    # Detailed README with setup instructions and project explanation
└──  requirements.txt             # List of Python dependencies
```

## MuseTalk: Real-Time Lip Sync with High-Quality Talking Avatars

MuseTalk is an open-source project that provides a real-time lip-sync solution with high-quality talking avatars. It was developed by the [Tencent Music Entertainment Lyra Lab](https://huggingface.co/TMElyralab) in April 2024. As of late 2024, it's considered state-of-the-art in terms of openly available zero-shot lipsyncing models. It's also available under the MIT License, which makes it usable both academically and commercially. 

### How does it work?

The technical report of MuseTalk is not yet available. However, I have a shallow understanding of how it works. MuseTalk is able to modify an unseen face according to a provided audio with a face region of 256 x 256. 

It uses Whisper-tiny's audio features to perform the facial modifications. The architecture of the generation network is borrowed from the UNet of __stable-diffusion-v1-4__ where audio embeddings were fused with the image embeddings using cross-attention.

![](assets/musetalk_arc.jpg)

Based on the project structures and descriptions in the sources, we can infer the following about MuseTalk's likely involvement in the lip-syncing pipeline:

1. __Input:__ The system takes an input image or video of a person's face and an audio track as input.

2. __Preprocessing:__ 
    - __Face Detection and Parsing:__ Models like dwpose and face-parse-bisent are used to detect the face in the input image/video and segment facial features (eyes, nose, mouth).
    - __Audio Encoding:__ The audio input is likely converted into a suitable representation, possibly using models like Whisper to extract phonetic features or directly using raw audio features.
    - __VAE Encoding:__ A Stable Diffusion Variational Autoencoder (VAE), potentially the "sd-vae-ft-mse," might be used to encode the input face image into a latent space representation. 

3. __MuseTalk Lip Syncing:__
    - __Latent Space Modification:__ MuseTalk operates in the latent space of the VAE, meaning it modifies the encoded facial features based on the input audio features.
    - __UNet Architecture:__ MuseTalk borrows the UNet architecture from Stable Diffusion for its generation network. This UNet likely takes the latent face representation and audio embeddings as input and outputs a modified latent face representation with synchronized lip movements.
    - __Single-Step Inpainting:__ Unlike diffusion models, MuseTalk performs lip-syncing as a single-step inpainting process in the latent space. This suggests an efficient and potentially real-time capable approach.

4. __Postprocessing:__
    - __VAE Decoding:__ The modified latent face representation is decoded back into an image using the VAE decoder.
    - __Blending and Output Generation:__ The generated lip movements are blended with the original input image/video frame to create the final lip-synced output.

## Getting Started

### Installation

To prepare the Python environment and install additional packages such as opencv, diffusers, mmcv, etc., please follow the steps below:

#### Build the environment
Conda is recommended for creating the environment. You can create a new environment using the following command:
```bash
conda create -n rtlipsync python=3.10
conda activate rtlipsync
```

```bash
pip install -r requirements.txt
pip install -e . 
conda install pyaudio
```

#### mmlab packages

```bash
pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0" 
```

#### Download ffmpeg-static

Please follow the instructions [here](https://www.johnvansickle.com/ffmpeg/faq/) to download the ffmpeg-static package.

```bash
# https://www.johnvansickle.com/ffmpeg/old-releases/
wget https://www.johnvansickle.com/ffmpeg/old-releases/ffmpeg-4.4-amd64-static.tar.xz

tar -xvf ffmpeg-4.4-amd64-static.tar.xz

export FFMPEG_PATH=./ffmpeg-git-20240629-amd64-static
```

### Download weights

Run the following command to download the weights of the model:

```bash
python scripts/download_weights.py
```

Weights include the following models:
- [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- [whisper](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)
- [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
- [face-parse-bisent](https://github.com/zllrunning/face-parsing.PyTorch)
- [resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth)

Finally, these weights should be organized in models as follows:

```
rtlipsync/models/
├── musetalk
│   └── musetalk.json
│   └── pytorch_model.bin
├── dwpose
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── sd-vae-ft-mse
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── whisper
    └── tiny.pt
```

## Quick Start

### Inference

```bash
python -m rtlipsync.inference.inference --inference_config configs/inference/test.yaml 
```

`configs/inference/test.yaml` is the path to the inference configuration file, including `video_path` and `audio_path`. The `video_path` should be either a video file, an image file or a directory of images.

You are recommended to input video with 25fps, the same fps used when training the model. If your video is far less than 25fps, you are recommended to apply frame interpolation or directly convert the video to 25fps using ffmpeg.

__Use of bbox_shift to have adjustable results__

We have found that upper-bound of the mask has an important impact on mouth openness. Thus, to control the mask region, we suggest using the `bbox_shift` parameter. Positive values (moving towards the lower half) increase mouth openness, while negative values (moving towards the upper half) decrease mouth openness.

You can start by running with the default configuration to obtain the adjustable value range, and then re-run the script within this range.

For example, in the case of `Xinying Sun`, after running the default configuration, it shows that the adjustable value rage is [-9, 9]. Then, to decrease the mouth openness, we set the value to be -7.

```bash
python -m rtlipsync.inference.inference --inference_config configs/inference/test.yaml --bbox_shift -7 
```

### Real-time Inference

```bash
python -m rtlipsync.inference.realtime_inference --inference_config configs/inference/realtime.yaml --batch_size 4
```

This script first supports real-time inference by applying necessary pre-processing such as face detection, face parsing and VAE encode in advance. During inference, only UNet and the VAE decoder are involved, which makes MuseTalk real-time.

`configs/inference/realtime.yaml` is the path to the real-time inference configuration file, including `preparation`, `video_path` , `bbox_shift` and `audio_clips`.

1. Set `preparation` to True in `realtime.yaml` to prepare the materials for a new avatar. (If the `bbox_shift` has changed, you also need to re-prepare the materials.)
2. After that, the `avatar` will use an audio clip selected from `audio_clips` to generate video.
```bash
Inferring using: data/audio/yongen.wav
```
3. While MuseTalk is inferring, sub-threads can simultaneously stream the results to the users. The generation process can achieve 30fps+ on an NVIDIA Tesla V100.
4. Set `preparation` to `False` and run this script if you want to generate more videos using the same avatar.

Note for Real-time inference
1. If you want to generate multiple videos using the same avatar/video, you can also use this script to __SIGNIFICANTLY__ expedite the generation process.
2. In the previous script, the generation time is also limited by I/O (e.g. saving images). If you just want to test the generation speed without saving the images, you can run

```bash
python -m rtlipsync.inference.realtime_inference --inference_config configs/inference/realtime.yaml --skip_save_images
```

### Real-time Inference API

The FastAPI application allows you to generate digital human videos and preprocess avatar data. Below are the steps to interact with the API.

#### Start the FastAPI application

To run the FastAPI server locally:

```bash
uvicorn main:app --port 8010 --reload
```

#### FastAPI Endpoints

__Old API endpoints:__

1. __Check the status of the server:__
   - `GET /digital_human/check`
   - __Response__: 
   ```json
   {"status": "Server is running"}
   ```

2. __Generate Digital Human Video:__
   - `POST /digital_human/gen`
   - __Request Body__:
   ```json
   {
     "user_id": "user123",
     "request_id": "req123", 
     "streamer_id": "streamer1",
     "tts_path": "data/audio/yongen.wav",
     "chunk_id": 0
   }
   ```
   - __Response__:
   ```json
   {
     "user_id": "user123",
     "request_id": "req123",
     "digital_human_mp4_path": "/path/to/video.mp4"
   }
   ```

3. __Prepare Avatar Data:__
   - `POST /digital_human/preprocess`
   - __Request Body__:
   ```json
   {
     "user_id": "user123",
     "request_id": "req123",
     "streamer_id": "streamer1",
     "video_path": "/path/to/video.mp4"
   }
   ```
   - __Response__:
   ```json
   {
     "user_id": "user123",
     "request_id": "req123"
   }
   ```

__New API endpoints:__

Here's how to use the API for real-time avatar voice imitation and audio file handling.

### 1. **Setup and Run the API**

First, ensure you have installed the necessary dependencies and have your environment set up with the required libraries (`fastapi`, `uvicorn`, `aiortc`, etc.).

Run the FastAPI server with this command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8010
```

### 2. **Endpoints Overview**

| Endpoint          | Method    | Description                                                   |
|-------------------|-----------|---------------------------------------------------------------|
| `/humanecho`      | WebSocket | Real-time WebSocket connection for streaming voice/text to the avatar. |
| `/offer`          | POST      | WebRTC offer for setting up peer connections for video/audio streaming. |
| `/humanaudio`     | POST      | Upload audio files for the avatar to imitate.                 |
| `/set_audiotype`  | POST      | Set the audio type for real-time audio imitation.             |
| `/record`         | POST      | Start or stop recording of the avatar's output video.         |
| `/is_speaking`    | POST      | Check if the avatar is currently speaking.                    |

### 3. **How to Use Each API Endpoint**

First, ensure you have `aiohttp` installed to handle HTTP requests and WebSocket communication:

```bash
pip install aiohttp
```

#### 3.1 **WebSocket for Real-time Voice/Text Imitation (`/humanecho`)**

Use WebSocket to stream real-time voice or text, which the avatar will imitate.

**Client Example**:

```python
import asyncio
import aiohttp

async def send_message_via_websocket():
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect('ws://localhost:8010/humanecho') as ws:
            print("WebSocket connection established!")
            
            # Send a text message
            await ws.send_str("Hello, Avatar!")
            
            # Receive response (if any)
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    print(f"Response from server: {msg.data}")
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break

# Run the WebSocket client
asyncio.run(send_message_via_websocket())
```

#### 3.2 **Upload Audio Files for Avatar to Imitate (`/humanaudio`)**

Send an audio file to the avatar for it to imitate. The avatar will sync its lips and movements with the audio provided.

**Request Example (cURL)**:

```bash
curl -X POST "http://<server-ip>:8010/humanaudio" \
    -F "sessionid=0" \
    -F "file=@/path/to/your/audiofile.wav"
```

**Parameters**:
- `sessionid`: The ID of the current session (e.g., `0` for the first session).
- `file`: The path to the audio file you want to upload.

```python
import aiohttp

async def upload_audio_file(session_id, file_path):
    async with aiohttp.ClientSession() as session:
        with open(file_path, 'rb') as file:
            audio_data = {'file': file}
            payload = {'sessionid': str(session_id)}
            form = aiohttp.FormData()
            form.add_field('sessionid', str(session_id))
            form.add_field('file', open(file_path, 'rb'), filename=file_path)
            
            async with session.post('http://localhost:8010/humanaudio', data=form) as response:
                response_json = await response.json()
                print(f"Response: {response_json}")

# Example usage
session_id = 0
file_path = '/path/to/your/audio.wav'
asyncio.run(upload_audio_file(session_id, file_path))
```

#### 3.3 **POST WebRTC Offer (`/offer`)**

This endpoint sets up the WebRTC peer connection to stream real-time video/audio. It's mainly used to establish a connection for streaming purposes (like using the avatar in real-time meetings or videos).

**Request Example (cURL)**:

```bash
curl -X POST "http://<server-ip>:8010/offer" \
    -H "Content-Type: application/json" \
    -d '{"sdp": "<your_sdp>", "type": "offer"}'
```

You will need to pass the SDP (Session Description Protocol) information from the WebRTC offer. This typically comes from WebRTC client libraries in web applications.

#### 3.4 **Set Audio Type (`/set_audiotype`)**

This endpoint allows you to change the audio type or settings for the avatar's audio processing.

**Request Example (cURL)**:

```bash
curl -X POST "http://<server-ip>:8010/set_audiotype" \
    -H "Content-Type: application/json" \
    -d '{
          "sessionid": 0,
          "audiotype": "stereo",
          "reinit": false
        }'
```

**Parameters**:
- `sessionid`: The ID of the session (e.g., `0` for the first session).
- `audiotype`: The type of audio processing (e.g., `"stereo"`).
- `reinit`: Whether to reinitialize the audio settings (`true` or `false`).

```python
import aiohttp

async def set_audio_type(session_id, audio_type, reinit=False):
    payload = {
        'sessionid': session_id,
        'audiotype': audio_type,
        'reinit': reinit
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post('http://localhost:8010/set_audiotype', json=payload) as response:
            result = await response.json()
            print(f"Audio type set response: {result}")

# Example usage
asyncio.run(set_audio_type(0, 'stereo', reinit=True))
```

#### 3.5 **Record the Avatar's Video (`/record`)**

Start or stop recording the avatar's video imitation.

**Request Example (cURL)**:

```bash
curl -X POST "http://<server-ip>:8010/record" \
    -H "Content-Type: application/json" \
    -d '{
          "sessionid": 0,
          "type": "start_record"
        }'
```

- `type`: Use `"start_record"` to start recording and `"end_record"` to stop recording.

```python
import aiohttp

async def record_avatar(session_id, action_type):
    payload = {
        'sessionid': session_id,
        'type': action_type  # either 'start_record' or 'end_record'
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post('http://localhost:8010/record', json=payload) as response:
            result = await response.json()
            print(f"Record action response: {result}")

# Example usage to start recording
asyncio.run(record_avatar(0, 'start_record'))

# Example usage to stop recording
asyncio.run(record_avatar(0, 'end_record'))
```

#### 3.6 **Check if the Avatar is Speaking (`/is_speaking`)**

Check if the avatar is currently speaking based on the audio being processed.

**Request Example (cURL)**:

```bash
curl -X POST "http://<server-ip>:8010/is_speaking" \
    -H "Content-Type: application/json" \
    -d '{
          "sessionid": 0
        }'
```

**Response**:
- Returns whether the avatar is speaking (e.g., `true` or `false`).

```python
import aiohttp

async def is_avatar_speaking(session_id):
    payload = {
        'sessionid': session_id
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post('http://localhost:8010/is_speaking', json=payload) as response:
            result = await response.json()
            print(f"Is avatar speaking: {result['data']}")

# Example usage
asyncio.run(is_avatar_speaking(0))
```

### 4. **Real-time Interaction Workflow**

1. **Real-time Voice/Text Imitation**:
   - Establish a WebSocket connection to `/humanecho`.
   - Send voice or text data through WebSocket.
   - The avatar will imitate the received text or audio.

2. **Audio File Upload**:
   - Send audio files to the `/humanaudio` endpoint.
   - The avatar will sync its lips/movements based on the provided audio.

3. **WebRTC Setup**:
   - If you want to stream video/audio to and from the avatar, use `/offer` to establish a WebRTC connection.

4. **Control Audio Settings**:
   - Use `/set_audiotype` to change how the avatar handles audio.
   
5. **Recording**:
   - Start and stop recording the avatar's output via `/record`.

6. **Checking Avatar State**:
   - Use `/is_speaking` to check if the avatar is actively speaking.

### 5. **Typical Use Case Example**

Let’s say you want to upload an audio file for the avatar to imitate:

- First, ensure your server is running on `http://localhost:8010`.
- You can upload an audio file using the following cURL command:

```bash
curl -X POST "http://localhost:8010/humanaudio" \
    -F "sessionid=0" \
    -F "file=@/path/to/audio.wav"
```

- After the file is processed, the avatar will synchronize its lips and movements to match the audio.

For real-time interaction via WebSocket:

- You can open a WebSocket connection to `ws://localhost:8010/humanecho` and stream live text or audio.
  
<!-- ### Conclusion

This API provides a flexible way to interact with an avatar that can mimic the voice/audio you provide. You can either use pre-recorded audio files or stream real-time voice input. The WebRTC setup allows you to stream video and audio, making it suitable for real-time video conferencing applications. -->

### References
1. MuseTalk: Real-Time Lip Sync with High-Quality Talking Avatars. [GitHub](https://github.com/TMElyralab/MuseTalk)
2. HeyGen: https://docs.heygen.com/docs/quick-start
3. metahuman-stream: Real time interactive streaming digital human. [GitHub](https://github.com/lipku/metahuman-stream)
4. Streamer-Sales: 销冠 —— 卖货主播大模型. [GitHub](https://github.com/PeterH0323/Streamer-Sales)

