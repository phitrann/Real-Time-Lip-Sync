import time
import numpy as np

import queue
from queue import Queue
import multiprocessing as mp
from rtlipsync.whisper.audio2feature import Audio2Feature
from rtlipsync.utils.utils import load_audio_model


class BaseASR:
    def __init__(self, opt, parent=None):
        self.opt = opt
        self.parent = parent

        self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.queue = Queue()
        self.output_queue = mp.Queue()

        self.batch_size = opt.batch_size

        self.frames = []
        self.stride_left_size = opt.l
        self.stride_right_size = opt.r
        #self.context_size = 10
        self.feat_queue = mp.Queue(2)

        #self.warm_up()

    def pause_talk(self):
        self.queue.queue.clear()

    def put_audio_frame(self, audio_chunk):  # 16khz 20ms pcm
        # Convert audio_chunk from bytes to NumPy array
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        self.queue.put(audio_array)

    def get_audio_frame(self):        
        try:
            frame = self.queue.get(block=True,timeout=0.01)
            type = 0
            #print(f'[INFO] get frame {frame.shape}')
        except queue.Empty:
            if self.parent and self.parent.curr_state>1: #播放自定义音频
                frame = self.parent.get_audio_stream(self.parent.curr_state)
                type = self.parent.curr_state
            else:
                frame = np.zeros(self.chunk, dtype=np.float32)
                type = 1

        return frame,type 

    def is_audio_frame_empty(self)->bool:
        return self.queue.empty()

    def get_audio_out(self):  #get origin audio pcm to nerf
        return self.output_queue.get()
    
    def warm_up(self):
        for _ in range(self.stride_left_size + self.stride_right_size):
            audio_frame,type=self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame,type))
        for _ in range(self.stride_left_size):
            self.output_queue.get()

    def run_step(self):
        pass

    def get_next_feat(self,block,timeout):        
        return self.feat_queue.get(block,timeout)

class MuseASR(BaseASR):
    def __init__(self, opt, parent, audio_processor: Audio2Feature):
        super().__init__(opt,parent)
        self.audio_processor = audio_processor

    def run_step(self):
        ############################################## extract audio feature ##############################################
        if self.audio_processor is None:
            self.audio_processor = load_audio_model()
        start_time = time.time()
        
        for _ in range(self.batch_size * 2):
            audio_frame, frame_type = self.get_audio_frame()
            if audio_frame is not None and audio_frame.size > 0:
                # Append valid audio frames
                self.frames.append(audio_frame)
                self.output_queue.put((audio_frame, frame_type))
            else:
                print("Received empty or invalid audio frame, appending silence.")
                # Append a zero-filled frame to maintain timing consistency
                zero_frame = np.zeros(self.chunk, dtype=np.float32)
                self.frames.append(zero_frame)
                self.output_queue.put((zero_frame, frame_type))
        
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        # Filter out invalid frames before concatenation
        valid_frames = [frame for frame in self.frames if frame is not None and frame.size > 0]
        if not valid_frames:
            print("No valid audio frames to process, skipping run_step.")
            return

        inputs = np.concatenate(valid_frames)  # [N * chunk]
        whisper_feature = self.audio_processor.audio2feat(inputs)
        
        print(f"Processing audio took {(time.time() - start_time) * 1000:.2f} ms, inputs shape: {inputs.shape}, whisper_feature length: {len(whisper_feature)}")
        
        whisper_chunks = self.audio_processor.feature2chunks(
            feature_array=whisper_feature,
            fps=self.fps / 2,
            batch_size=self.batch_size,
            start=self.stride_left_size / 2
        )
        
        self.feat_queue.put(whisper_chunks)
        # Discard the old frames to save memory
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]
