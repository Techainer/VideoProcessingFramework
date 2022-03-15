import asyncio
import collections
import json
import os
import ssl

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from loguru import logger

ROOT = os.path.dirname(__file__)



class DeQueueVideoStreamTrack(VideoStreamTrack):
    """
    A video track that returns frames from a dequeue.
    """
    def __init__(self, dequeue: collections.deque):
        super().__init__()
        self.dequeue = dequeue
        if self.dequeue.maxlen is None:
            logger.warning("dequeue for WebRTC stream should have a maxlen arround 30 frames, \
                don't have maxlen set might cause memory leak when there are no clients connected to consume the stream")
        elif self.dequeue.maxlen > 30:
            logger.warning("dequeue for WebRTC stream should have a maxlen arround 30 frames, \
                have maxlen set greater than it might make the stream be delayed")

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        while True:
            try:
                frame = self.dequeue.pop()
            except IndexError:
                await asyncio.sleep(1/100)
                continue
            frame = VideoFrame.from_ndarray(frame, format="bgr24")
            frame.pts = pts
            frame.time_base = time_base
            return frame

async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

class WebRTCPreview:
    def __init__(self,
                 host,
                 port,
                 frame_dequeue: collections.deque,
                 cert_file: str = None,
                 key_file: str = None):
        self.pcs = set()
        self.app = web.Application()
        self.app.on_shutdown.append(self.on_shutdown)
        self.app.router.add_get("/", index)
        self.app.router.add_get("/client.js", javascript)
        self.app.router.add_post("/offer", self.offer)
        self.frame_dequeue = frame_dequeue
        self.host = host
        self.port = port
        if cert_file is not None:
            ssl_context = ssl.SSLContext()
            ssl_context.load_cert_chain(cert_file, key_file)
        else:
            ssl_context = None
        self.ssl_context = ssl_context

    async def start(self):
        # set up the web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host=self.host, port=self.port, ssl_context=self.ssl_context)
        await site.start()
        names = sorted(str(s.name) for s in runner.sites)
        print(
            "======== Running on {} ========\n"
            "(Press CTRL+C to quit)".format(", ".join(names))
        )
        delay = 60
        while True:
            await asyncio.sleep(delay)

    async def on_shutdown(self, app):
        # close peer connections
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()

    async def offer(self, request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print("Connection state is %s" % pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        await pc.setRemoteDescription(offer)
        pc.addTrack(DeQueueVideoStreamTrack(self.frame_dequeue))

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )
