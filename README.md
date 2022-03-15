# VideoProcessingFramework
This is a fork from the original [VPF](https://github.com/NVIDIA/VideoProcessingFramework). You can find more information there. 
The purpose of this repo is to create a developing environment. 

## Prerequisite
- Download [Video Codec SDK 9.1.23](https://developer.nvidia.com/video-codec-sdk-archive) and unzip.

## Docker Instructions (Linux)
- Export env var:
```
source scripts/export.sh
```
- Build image:
```
sh scripts/build.sh
```
- Mount and start developing.
```
sh scripts/run.sh
```

## Testing
You have to host a Janus server first (to stream using WebRTC).
- Decode rtsp stream using ffmpeg. Reconfig `DecodeRTSP_ffmpeg.py` if needed:
```
python DecodeRTSP_ffmpeg.py
```
- Decode rtsp stream w/o using ffmpeg. Reconfig `DecodeRTSP_wo_ffmpeg.py` if needed:
```
python DecodeRTSP_wo_ffmpeg.py
```