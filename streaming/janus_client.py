import asyncio
import random
import string
import time
import traceback
from collections import deque
from threading import Thread

import aiohttp
import cv2
import requests
from aiortc import (RTCConfiguration, RTCIceServer, RTCPeerConnection,
                    RTCSessionDescription, VideoStreamTrack)
from loguru import logger

from .webrtc_streamer import DeQueueVideoStreamTrack


def transaction_id():
    return "".join(random.choice(string.ascii_letters) for x in range(12))


class JanusPlugin:
    def __init__(self, session, url):
        self._queue = asyncio.Queue()
        self._session = session
        self._url = url

    async def send(self, payload):
        message = {"janus": "message", "transaction": transaction_id()}
        message.update(payload)
        async with self._session._http.post(self._url, json=message) as response:
            data = await response.json()
            assert data["janus"] == "ack"
        
        response = await self._queue.get()
        assert response["transaction"] == message["transaction"]
        return response
    
class JanusSession:
    def __init__(self, url):
        self._http = None
        self._poll_task = None
        self._plugins = {}
        self._root_url = url
        self._session_url = None

    async def attach(self, plugin_name: str) -> JanusPlugin:
        message = {
            "janus": "attach",
            "plugin": plugin_name,
            "transaction": transaction_id(),
        }
        async with self._http.post(self._session_url, json=message) as response:
            data = await response.json()
            assert data["janus"] == "success"
            plugin_id = data["data"]["id"]
            plugin = JanusPlugin(self, self._session_url + "/" + str(plugin_id))
            self._plugins[plugin_id] = plugin
            return plugin

    async def create(self):
        self._http = aiohttp.ClientSession()
        message = {"janus": "create", "transaction": transaction_id()}
        async with self._http.post(self._root_url, json=message) as response:
            data = await response.json()
            assert data["janus"] == "success"
            session_id = data["data"]["id"]
            self._session_url = self._root_url + "/" + str(session_id)

        self._poll_task = asyncio.ensure_future(self._poll())

    async def destroy(self):
        if self._poll_task:
            self._poll_task.cancel()
            self._poll_task = None

        if self._session_url:
            message = {"janus": "destroy", "transaction": transaction_id()}
            async with self._http.post(self._session_url, json=message) as response:
                data = await response.json()
                assert data["janus"] == "success"
            self._session_url = None

        if self._http:
            await self._http.close()
            self._http = None

    async def _poll(self):
        while True:
            params = {"maxev": 1, "rid": int(time.time() * 1000)}
            async with self._http.get(self._session_url, params=params) as response:
                data = await response.json()
                if data["janus"] == "event":
                    plugin = self._plugins.get(data["sender"], None)
                    if plugin:
                        await plugin._queue.put(data)
                    else:
                        logger.debug(f"Received event data form Janus: {data}")


class JanusClient:
    def __init__(self, 
                janus_server_url: str = "http://techainer-rtx3090-2.localdomain:8088/janus",
                ice_server_url: str = "turn:janus.truongkyle.tech:3478?transport=udp",
                ice_server_username: str = None,
                ice_server_password: str = None,
                frame_dequeue: deque = None,
                room: int = 1234,
                cam_id: int = 1,
                retry: int = 3):
        self.janus_server_url = janus_server_url
        self.ice_server_url = ice_server_url
        self.ice_server_username = ice_server_username
        self.ice_server_password = ice_server_password
        self.frame_dequeue = frame_dequeue
        self.room = room
        self.cam_id = cam_id
        counter = 0
        while True:
            self.session = JanusSession(janus_server_url)
            self.pcs = set()
            self.running = True
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            self.done_event = asyncio.Event()
            self.error_event = asyncio.Event()
            self.main_thread = Thread(target=self.asyncio_thread, args=())
            self.main_thread.start()
            try:
                self.ensure_publishing()
            except:
                self.stop()
                counter += 1
                if counter > retry:
                    logger.error("Try to init Janus client for {} times".format(counter))
                    raise Exception("Failed to start Janus client")
                else:
                    continue
            break

    def ensure_publishing(self):
        while not self.done_event.is_set():
            if self.error_event.is_set():
                raise Exception("Error initializing Janus client")
            time.sleep(1/10000)

    def asyncio_thread(self):
        self.loop.run_until_complete(
            self.run()
        )

    def stop(self):
        self.running = False
        if self.main_thread.is_alive():
            self.main_thread.join()
        logger.info("Done joining main thread")

        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.session.destroy())
        logger.info("Done destroying Janus session")

        # close peer connections
        coros = [pc.close() for pc in self.pcs]
        self.loop.run_until_complete(asyncio.gather(*coros))
        logger.info("Done closing peer connections")

    async def run(self):
        try:
            await self.session.create()

            # join video room
            plugin = await self.session.attach("janus.plugin.videoroom")
            _ = await plugin.send(
                {
                    "body": {
                        "display": str(self.cam_id),
                        "ptype": "publisher",
                        "request": "join",
                        "room": self.room,
                    }
                }
            )
            await self.publish(plugin)
        except:
            logger.error(f"Error when Janus client is warming up: {traceback.format_exc()}")
            self.error_event.set()
            return
        self.done_event.set()

        while self.running:
            await asyncio.sleep(5)

    async def publish(self, plugin):
        """
        Send video to the room.
        """
        if self.ice_server_url is not None and self.ice_server_username is not None and self.ice_server_password is not None:
            ice_config = RTCConfiguration(
                iceServers=[
                    RTCIceServer(urls=self.ice_server_url, username=self.ice_server_username, credential=self.ice_server_password)
                ]
            )
            pc = RTCPeerConnection(configuration=ice_config)
        else:
            pc = RTCPeerConnection()
        self.pcs.add(pc)

        # configure media
        media = {"audio": False, "video": True}

        pc.addTrack(DeQueueVideoStreamTrack(self.frame_dequeue))

        # send offer
        await pc.setLocalDescription(await pc.createOffer())
        request = {"request": "configure"}
        request.update(media)
        logger.debug(f"Configured offer to Janus {request}")
        response = await plugin.send(
            {
                "body": request,
                "jsep": {
                    "sdp": pc.localDescription.sdp,
                    "trickle": False,
                    "type": pc.localDescription.type,
                },
            }
        )

        logger.debug(f"Response to configured offer from Janus {response}")

        # apply answer
        await pc.setRemoteDescription(
            RTCSessionDescription(
                sdp=response["jsep"]["sdp"], type=response["jsep"]["type"]
            )
        )
