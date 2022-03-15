import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from collections import deque
from io import BytesIO
from multiprocessing import Process
from threading import Thread
from typing import Dict

import cv2
import numpy as np
import pycuda.driver as cuda
import torch
from loguru import logger

import PyNvCodec as nvc
from streaming import JanusClient

if os.name == 'nt':
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    if cuda_path:
        os.add_dll_directory(cuda_path)
    else:
        print("CUDA_PATH environment variable is not set.", file=sys.stderr)
        print("Can't set CUDA DLLs search path.", file=sys.stderr)
        exit(1)

    # Add PATH as well for minor CUDA releases
    sys_path = os.environ["PATH"]
    if sys_path:
        paths = sys_path.split(';')
        for path in paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)
    else:
        print("PATH environment variable is not set.", file=sys.stderr)
        exit(1)

def get_stream_params(url: str) -> Dict:
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format', '-show_streams', url]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout = proc.communicate()[0]

    bio = BytesIO(stdout)
    json_out = json.load(bio)

    params = {}
    if not 'streams' in json_out:
        return {}

    for stream in json_out['streams']:
        if stream['codec_type'] == 'video':
            params['width'] = stream['width']
            params['height'] = stream['height']
            params['framerate'] = float(eval(stream['avg_frame_rate']))

            codec_name = stream['codec_name']
            is_h264 = True if codec_name == 'h264' else False
            is_hevc = True if codec_name == 'hevc' else False
            if not is_h264 and not is_hevc:
                raise ValueError("Unsupported codec: " + codec_name +
                                 '. Only H.264 and HEVC are supported in this sample.')
            else:
                params['codec'] = nvc.CudaVideoCodec.H264 if is_h264 else nvc.CudaVideoCodec.HEVC

                pix_fmt = stream['pix_fmt']
                is_yuv420 = pix_fmt == 'yuv420p'
                is_yuv444 = pix_fmt == 'yuv444p'

                # YUVJ420P and YUVJ444P are deprecated but still wide spread, so handle
                # them as well. They also indicate JPEG color range.
                is_yuvj420 = pix_fmt == 'yuvj420p'
                is_yuvj444 = pix_fmt == 'yuvj444p'

                if is_yuvj420:
                    is_yuv420 = True
                    params['color_range'] = nvc.ColorRange.JPEG
                if is_yuvj444:
                    is_yuv444 = True
                    params['color_range'] = nvc.ColorRange.JPEG

                if not is_yuv420 and not is_yuv444:
                    raise ValueError("Unsupported pixel format: " +
                                     pix_fmt +
                                     '. Only YUV420 and YUV444 are supported in this sample.')
                else:
                    params['format'] = nvc.PixelFormat.NV12 if is_yuv420 else nvc.PixelFormat.YUV444

                # Color range default option. We may have set when parsing
                # pixel format, so check first.
                if 'color_range' not in params:
                    params['color_range'] = nvc.ColorRange.MPEG
                # Check actual value.
                if 'color_range' in stream:
                    color_range = stream['color_range']
                    if color_range == 'pc' or color_range == 'jpeg':
                        params['color_range'] = nvc.ColorRange.JPEG

                # Color space default option:
                params['color_space'] = nvc.ColorSpace.BT_601
                # Check actual value.
                if 'color_space' in stream:
                    color_space = stream['color_space']
                    if color_space == 'bt709':
                        params['color_space'] = nvc.ColorSpace.BT_709

                return params
    return {}

def rtsp_client(url: str, name: str, gpuID: int, frame_deque: deque) -> None:
    try:
        janus_client = JanusClient(
            janus_server_url="http://192.168.40.5:30888/janus",
            # ice_server_url="turn:janus.truongkyle.tech:3478?transport=udp",
            # ice_server_username="horus",
            # ice_server_password="horus123@!",
            frame_dequeue=frame_deque,
            cam_id=666,
        )
    except:
        logger.error(f"Error connecting to Janus: {traceback.format_exc()}")
        sys.exit(1)

    # Get stream parameters
    params = get_stream_params(url)

    if not len(params):
        raise ValueError("Can not get " + url + ' streams params')

    w = params['width']
    h = params['height']
    f = params['format']
    c = params['codec']
    g = gpuID

    # Prepare ffmpeg arguments
    if nvc.CudaVideoCodec.H264 == c:
        codec_name = 'h264'
    elif nvc.CudaVideoCodec.HEVC == c:
        codec_name = 'hevc'
    bsf_name = codec_name + '_mp4toannexb,dump_extra=all'
    
    cmd = [
        'ffmpeg',       '-hide_banner',
        '-i',           url,
        '-c:v',         'copy',
        '-bsf:v',       bsf_name,
        '-f',           codec_name,
        'pipe:1'
    ]
    # Run ffmpeg in subprocess and redirect it's output to pipe
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    # Create HW decoder class
    nvdec = nvc.PyNvDecoder(w, h, f, c, g)

    # Amount of bytes we read from pipe first time.
    read_size = 4096
    # Total bytes read and total frames decded to get average data rate
    rt = 0
    fd = 0

    width, height = w, h

    # Initialize colorspace conversion chain
    if params['color_space'] != nvc.ColorSpace.BT_709:
        nvYuv = nvc.PySurfaceConverter(width, height, nvdec.Format(), nvc.PixelFormat.YUV420, gpuID)
    else:
        nvYuv = None

    if nvYuv:
        nvCvt = nvc.PySurfaceConverter(width, height, nvYuv.Format(), nvc.PixelFormat.RGB, gpuID)
    else:
        nvCvt = nvc.PySurfaceConverter(width, height, nvdec.Format(), nvc.PixelFormat.RGB, gpuID)
    
    surface_tensor = torch.zeros(height, width, 3, dtype=torch.uint8,
                                device=torch.device(f'cuda:{gpuID}'))

    cspace, crange = params['color_space'], params['color_range']
    if nvc.ColorSpace.UNSPEC == cspace:
        cspace = nvc.ColorSpace.BT_601
    if nvc.ColorRange.UDEF == crange:
        crange = nvc.ColorRange.MPEG
    cc_ctx = nvc.ColorspaceConversionContext(cspace, crange)

    st_all = time.time()
    while True:
        try:
            # Pipe read underflow protection
            if not read_size:
                read_size = int(rt / fd)
                # Counter overflow protection
                rt = read_size
                fd = 1

            # Read data.
            # Amount doesn't really matter, will be updated later on during decode.
            # st = time.time()
            bits = proc.stdout.read(read_size)
            # logger.debug(f"Read bits from pipe took {(time.time() - st)*1000} ms")
            if not len(bits):
                print("Can't read data from pipe")
                break
            else:
                rt += len(bits)

            # Decode
            # st = time.time()
            enc_packet = np.frombuffer(buffer=bits, dtype=np.uint8)
            pkt_data = nvc.PacketData()
            # logger.debug(f"Decode package data took {(time.time() - st)*1000} ms")
            try:
                # st = time.time()
                surf = nvdec.DecodeSurfaceFromPacket(enc_packet, pkt_data)
                # surf = nvdec.DecodeSingleSurface()
                # logger.debug(f"Decode surface took {(time.time() - st)*1000} ms")

                if not surf.Empty():
                    fd += 1
                    # Shifts towards underflow to avoid increasing vRAM consumption.
                    if pkt_data.bsl < read_size:
                        read_size = pkt_data.bsl

                    # st = time.time()
                    if nvYuv:
                        yuvSurface = nvYuv.Execute(surf, cc_ctx)
                        cvtSurface = nvCvt.Execute(yuvSurface, cc_ctx)
                    else:
                        cvtSurface = nvCvt.Execute(surf, cc_ctx)
                    # logger.debug(f"Convert surface took {(time.time() - st)*1000} ms")

                    # st = time.time()
                    cvtSurface.PlanePtr().Export(surface_tensor.data_ptr(), width*3, gpuID)
                    # logger.debug(f"Export surface took {(time.time() - st)*1000} ms")

                    # logger.debug(f"Whole decoding flow took {(time.time() - st_all)*1000} ms")
                    logger.debug(f"Framerate: {fd/(time.time() - st_all)}")

                    img = cv2.cvtColor(surface_tensor.cpu().numpy(), cv2.COLOR_RGB2BGR)
                    frame_deque.appendleft(img)

            # Handle HW exceptions in simplest possible way by decoder respawn
            except nvc.HwResetException:
                st = time.time()
                nvdec = nvc.PyNvDecoder(w, h, f, c, g)
                logger.warning(f"HW reset: reset time = {(time.time() - st)*1000} ms")
                continue
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")
            janus_client.stop()
            return


if __name__ == "__main__":
    print("This sample decodes multiple videos in parallel on given GPU.")
    print("Input rtsp stream will be decoded to np.array.")
    print("Usage: SampleDecodeRTSP.py $gpu_id $url1 ... $urlN .")

    if(len(sys.argv) < 3):
        sys.argv.append(0)
        # sys.argv.append("rtsp://admin:Techainer123@192.168.50.5:554/media/video1")
        sys.argv.append("rtsp://admin:Techainer123@techainer-hikvision-office-2:554/media/video1")

    gpuID = int(sys.argv[1])
    urls = []

    for i in range(2, len(sys.argv)):
        urls.append(sys.argv[i])

    # urls = urls * 4

    pool = []
    for idx, url in enumerate(urls):
        frame_deque = deque(maxlen=10)
        client = Thread(target=rtsp_client, args=(
                url, str(idx), gpuID, frame_deque
            )
        )
        client.start()
        pool.append(client)

    for client in pool:
        client.join()

    