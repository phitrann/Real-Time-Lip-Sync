# Real-Time-Lip-Sync


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
Here, we provide the inference script. This script first applies necessary pre-processing such as face detection, face parsing and VAE encode in advance. During inference, only UNet and the VAE decoder are involved, which makes MuseTalk real-time.

```bash
python -m rtlipsync.inference.realtime_inference --inference_config configs/inference/realtime.yaml --batch_size 4
```

`configs/inference/realtime.yaml` is the path to the real-time inference configuration file, including `preparation`, `video_path` , `bbox_shift` and `audio_clips`.

1. Set `preparation` to True in `realtime.yaml` to prepare the materials for a new avatar. (If the `bbox_shift` has changed, you also need to re-prepare the materials.)
2. After that, the `avatar` will use an audio clip selected from `audio_clips` to generate video.
```bash
Inferring using: data/audio/yongen.wav
```
3. While MuseTalk is inferring, sub-threads can simultaneously stream the results to the users. The generation process can achieve 30fps+ on an NVIDIA Tesla V100.
4. Set `preparation` to `False` and run this script if you want to genrate more videos using the same avatar.

Note for Real-time inference
1. If you want to generate multiple videos using the same avatar/video, you can also use this script to __SIGNIFICANTLY__ expedite the generation process.
2. In the previous script, the generation time is also limited by I/O (e.g. saving images). If you just want to test the generation speed without saving the images, you can run

```bash
python -m rtlipsync.inference.realtime_inference --inference_config configs/inference/realtime.yaml --skip_save_images
```


