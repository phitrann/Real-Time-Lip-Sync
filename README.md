# Real-Time-Lip-Sync

This project makes use of the MuseTalk model to perform real-time lip-syncing on a video and provides an API for generating digital human videos.

## MuseTalk: Real-Time Lip Sync with High-Quality Talking Avatars
MuseTalk is a open-source project that provides a real-time lip-sync solution with high-quality talking avatars. It was developed by the [Tencent Music Entertainment Lyra Lab](https://huggingface.co/TMElyralab) in April 2024. As of late 2024, it’s considered state-of-the-art in terms of openly available zero-shot lipsyncing models. It’s also available under the MIT License, which makes it usable both academically and commercially. 

### How does it work?
The technical report of MuseTalk is not yet available. However, I have a shallow understanding of how it works. MuseTalk is able to modify an unseen face according to a provided audio with a face region of 256 x 256. 

It uses Whisper-tiny's audio features to perform the facial modifications. The architecture of the generation network is borrowed from the UNet of __stable-diffusion-v1-4__ where audio embeddings were fused with the image embeddings using cross-attention.

![](assets/musetalk_arc.jpg)

## Getting Started
### Installation
To prepare the Python environment and install additional packages such as opencv, diffusers, mmcv, etc., please follow the steps below:


#### Build the environment
```bash
pip install -r requirements.txt
pip install -e . 
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

```
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
```bash
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
uvicorn app.main:app --reload
```

#### FastAPI Endpoints
1. __Check the status of the server:__
- `GET /digital_human/check`
- __Response__: 
```
{"status": "Server is running"}
```

2. __Generate Digital Human Video:__
- `POST /digital_human/gen`
- __Request Body__:
```json
{
  "user_id": "user123",
  "request_id": "req123", 
  "streamer_id": "streamer1", # unique id for the streamer
  "tts_path": "data/audio/yongen.wav", # path to the audio file
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

