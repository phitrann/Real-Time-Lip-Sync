# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @Time    : 2024.4.16
# # @Author  : HinGwenWong

# import requests
# import streamlit as st
# from PIL import Image


# # FastAPI Endpoint URL
# DIGITAL_HUMAN_API_URL = "http://localhost:8000/generate_digital_human"

# # Set up the Streamlit page configuration
# st.set_page_config(
#     page_title="Digital Human Generator",
#     page_icon="🤖",
#     layout="wide",
# )

# # Function to call the FastAPI digital human generation endpoint
# def generate_digital_human(text_input, image_file):
#     files = {'file': image_file}
#     data = {'text_input': text_input}
    
#     # Make the API call to FastAPI
#     response = requests.post(DIGITAL_HUMAN_API_URL, data=data, files=files)
    
#     if response.status_code == 200:
#         st.success("Digital human generation successful!")
#         return response.content  # Assuming this returns the generated video file
#     else:
#         st.error(f"Failed to generate digital human: {response.text}")
#         return None

# # Streamlit UI
# st.title("Digital Human Generation")

# # Text input for the script the digital human will speak
# text_input = st.text_area("Enter the script for the digital human to say", "")

# # File upload for the avatar or image representing the digital human
# image_file = st.file_uploader("Upload the avatar image", type=["jpg", "jpeg", "png"])

# if st.button("Generate Digital Human"):
#     if text_input and image_file:
#         st.info("Generating digital human...")
#         result = generate_digital_human(text_input, image_file)
        
#         if result:
#             # Display the generated video
#             st.video(result)
#     else:
#         st.error("Please provide both a script and an avatar image.")


import asyncio
import websockets
import json
import aiohttp
import pyaudio
import wave
import av
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer
from aiortc import MediaStreamTrack

import os

os.environ["no_proxy"] = "localhost,172.16.87.75,127.0.0.1"


# class AudioStreamTrack(MediaStreamTrack):
#     kind = "audio"

#     def __init__(self, track):
#         super().__init__()
#         self.track = track

#     async def recv(self):
#         frame = await self.track.recv()
#         return frame

# async def send_offer(session, url, max_retries=5, retry_delay=5):
#     for attempt in range(max_retries):
#         try:
#             pc = RTCPeerConnection()
            
#             # Add audio track
#             audio_track = AudioStreamTrack(MediaPlayer('data/audio/elon.wav').audio)
#             pc.addTrack(audio_track)

#             # Create and set local description
#             await pc.setLocalDescription(await pc.createOffer())

#             # Send the offer to the server
#             async with session.post(f'{url}/offer', json={
#                 'sdp': pc.localDescription.sdp,
#                 'type': pc.localDescription.type
#             }) as response:
#                 if response.status == 400:
#                     error_msg = await response.json()
#                     if "Max sessions reached" in error_msg.get('error', ''):
#                         print(f"Max sessions reached. Retrying in {retry_delay} seconds...")
#                         await asyncio.sleep(retry_delay)
#                         continue
#                 response.raise_for_status()
#                 answer = await response.json()
#                 await pc.setRemoteDescription(RTCSessionDescription(sdp=answer['sdp'], type=answer['type']))
#                 return answer['sessionid'], pc
#         except aiohttp.ClientError as e:
#             print(f"Connection error: {e}. Retrying in {retry_delay} seconds...")
#             await asyncio.sleep(retry_delay)
    
#     raise Exception("Failed to establish connection after maximum retries")
        

# async def send_audio_data(websocket, sessionid, audio_data):
#     await websocket.send(json.dumps({
#         'type': 'audio',
#         'sessionid': sessionid,
#         'data': audio_data.tobytes().decode('latin-1')  # Convert bytes to string
#     }))

# async def receive_video(pc, output_file):
#     # Set up a video recorder
#     recorder = av.open(output_file, 'w')
#     video_stream = recorder.add_stream('h264', rate=30)
    
#     @pc.on("track")
#     async def on_track(track):
#         if track.kind == "video":
#             while True:
#                 frame = await track.recv()
#                 # Encode and write the frame
#                 packet = video_stream.encode(frame)
#                 if packet:
#                     recorder.mux(packet)

# async def main():
#     url = "http://localhost:8010"  # Replace with your server's address
#     ws_url = "ws://localhost:8010/ws"  # WebSocket URL

#     async with aiohttp.ClientSession() as session:
#         try:
#             # Establish WebRTC connection with retry logic
#             sessionid, pc = await send_offer(session, url)

#             # Set up WebSocket connection
#             async with websockets.connect(ws_url) as websocket:
#                 # Set up PyAudio for microphone input
#                 p = pyaudio.PyAudio()
#                 stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

#                 # Start receiving video
#                 video_task = asyncio.create_task(receive_video(pc, 'output_video.mp4'))

#                 try:
#                     while True:
#                         # Read audio data from microphone
#                         audio_data = stream.read(1024)
#                         # Send audio data through WebSocket
#                         await send_audio_data(websocket, sessionid, audio_data)
#                 except KeyboardInterrupt:
#                     print("Stopping...")
#                 finally:
#                     # Clean up
#                     stream.stop_stream()
#                     stream.close()
#                     p.terminate()
#                     await pc.close()
#                     await video_task
#         except Exception as e:
#             print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     asyncio.run(main())

#########################################################

# import aiohttp

# async def upload_audio_file(session_id, file_path):
#     async with aiohttp.ClientSession() as session:
#         with open(file_path, 'rb') as file:
#             audio_data = {'file': file}
#             payload = {'sessionid': str(session_id)}
#             form = aiohttp.FormData()
#             form.add_field('sessionid', str(session_id))
#             form.add_field('file', open(file_path, 'rb'), filename=file_path)
            
#             async with session.post('http://localhost:8010/humanaudio', data=form) as response:
#                 response_json = await response.json()
#                 print(f"Response: {response_json}")

# # Example usage
# session_id = 0
# file_path = 'data/audio/elon.wav'
# asyncio.run(upload_audio_file(session_id, file_path))
 
 #############################################################
 
# import asyncio
# import wave
# from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
# from aiortc.contrib.media import MediaStreamError
# from pydub import AudioSegment
# from pydub.utils import make_chunks
# import aiohttp

# # Custom MediaStreamTrack to stream audio from a file
# class AudioFileTrack(MediaStreamTrack):
#     kind = "audio"

#     def __init__(self, path):
#         super().__init__()  # Initialize the base class
#         self.audio = AudioSegment.from_file(path, format="wav")
#         self.chunks = make_chunks(self.audio, 20)  # 20ms chunks for WebRTC
#         self.chunk_index = 0

#     async def recv(self):
#         if self.chunk_index >= len(self.chunks):
#             raise MediaStreamError("End of stream")

#         # Get the next audio chunk (frame)
#         chunk = self.chunks[self.chunk_index]
#         self.chunk_index += 1

#         return chunk

# async def create_webrtc_offer(audio_path):
#     # Create a new RTCPeerConnection
#     pc = RTCPeerConnection()

#     # Add the audio track to the PeerConnection
#     audio_track = AudioFileTrack(audio_path)
#     pc.addTrack(audio_track)

#     # Create an SDP offer
#     offer = await pc.createOffer()
#     await pc.setLocalDescription(offer)

#     print("SDP Offer created:", offer.sdp)

#     # Send the offer to the FastAPI server (replace with your server address)
#     async with aiohttp.ClientSession() as session:
#         async with session.post('http://localhost:8010/offer', json={
#             'sdp': pc.localDescription.sdp,
#             'type': pc.localDescription.type
#         }) as response:
#             # Get the SDP answer from the server
#             answer = await response.json()
#             print("SDP Answer from server:", answer)

#             # Set the remote description
#             await pc.setRemoteDescription(RTCSessionDescription(sdp=answer['sdp'], type=answer['type']))

#             print("WebRTC connection established")

# # Run the WebRTC client and send audio
# audio_file_path = "data/audio/elon.wav"
# asyncio.run(create_webrtc_offer(audio_file_path))

########################################################
# import asyncio
# import websockets
# import json

# async def websocket_client(message: str, server_url: str):
#     """
#     WebSocket client to send a message and receive responses from the server.

#     Args:
#         message (str): The message to send to the server.
#         server_url (str): WebSocket server URL (e.g., ws://localhost:8000/humanecho).
#     """
#     try:
#         async with websockets.connect(server_url) as websocket:
#             # Send the message to the WebSocket server
#             print(f"Sending: {message}")
#             await websocket.send(message)
            
#             # Wait for the server's response
#             response = await websocket.recv()
#             print(f"Received: {response}")
            
#             # Optionally parse the response as JSON if needed
#             try:
#                 video_data = json.loads(response)
#                 print("Parsed video data:", video_data)
#             except json.JSONDecodeError:
#                 print("Response is not valid JSON:", response)

#     except websockets.exceptions.ConnectionClosed as e:
#         print(f"Connection closed: {e}")
#     except Exception as e:
#         print(f"Error: {e}")

# # Function to start the WebSocket client
# def start_websocket_client():
#     message = "test"  # You can replace this with dynamic input if needed
#     server_url = "ws://localhost:8010/humanecho"  # Replace with your WebSocket server URL
#     asyncio.run(websocket_client(message, server_url))

# start_websocket_client()

########################################################

import asyncio
import json
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription

async def negotiate(pc, url):
    await pc.setLocalDescription(await pc.createOffer())
    
    # Wait for ICE gathering to complete
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)
    
    offer = pc.localDescription
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, 
                                json={"sdp": offer.sdp, "type": offer.type},
                                headers={"Content-Type": "application/json"}) as response:
            answer = await response.json()
            
            await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
            
            return answer["sessionid"]

async def start_webrtc():
    config = {"sdpSemantics": "unified-plan"}
    # Uncomment the following line if you want to use STUN servers
    # config["iceServers"] = [{"urls": ["stun:stun.l.google.com:19302"]}]
    
    pc = RTCPeerConnection(config)
    
    # Add transceivers (equivalent to addTrack in the JS version)
    pc.addTransceiver("video", {"direction": "recvonly"})
    pc.addTransceiver("audio", {"direction": "recvonly"})
    
    # You would typically set up event listeners here for incoming tracks
    # For example:
    # @pc.on("track")
    # def on_track(track):
    #     if track.kind == "video":
    #         # Handle video track
    #     elif track.kind == "audio":
    #         # Handle audio track
    
    url = "http://localhost:8010/offer"  # Adjust this URL as needed
    session_id = await negotiate(pc, url)
    print(f"Session ID: {session_id}")
    
    # Keep the connection alive
    try:
        while True:
            await asyncio.sleep(1)
    finally:
        await pc.close()

if __name__ == "__main__":
    asyncio.run(start_webrtc())