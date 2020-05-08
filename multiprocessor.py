import threading
import cv2
from image_processor import ImageProcessor
from collections import deque 
from sound_player import SoundPlayer

class Task(object):
    def __init__(self, args, func):
        self.args = args
        self.func = func

class ProcessResult(object):
    def __init__(self):
        self.sounds = []
    
    def add_sound(self, sound):
        self.sounds.append(sound)


class Multiprocessor(object):
    def __init__(self):
        # self.task_holders = [TaskHolder() for i in range(3)]
        self.cap_to_process_to_queue = deque()
        self.process_to_display_queue = deque()
        self.img_processor = ImageProcessor()
        self.sound_player = SoundPlayer()

        # thread_capture = threading.Thread(target=self.capture, args=())
        # thread_process = threading.Thread(target=self.process, args=())
        # thread_display = threading.Thread(target=self.display, args=())

        # thread_cap_disp = threading.Thread(target=self.cap_display, args=())
        thread_process = threading.Thread(target=self.process, args=())

        thread_process.start()
        self.cap_display_independent()

        # thread_capture.start()
        # thread_process.start()
        # thread_display.start()
    
    def capture(self):
        cap = cv2.VideoCapture(0)
        while True:
            _, frame_orig = cap.read()
            frame_orig = self.img_processor.horizontal_flip(frame_orig)
            self.cap_to_process_to_queue.append(frame_orig)
    
    def process(self):
        ct = 0
        while True:
            
            ct += 1
            if ct % 100 != 0: continue
            # if len(self.cap_to_process_to_queue) == 0: continue
            # print('Got frame to process', len(self.cap_to_process_to_queue))
            # last_frame = self.cap_to_process_to_queue.popleft()
            self.sound_player.play_key_on_queue(ord('j'))
            # self.process_to_display_queue.append(last_frame)
    
    def display(self):
        while True:
            if len(self.process_to_display_queue) == 0: continue
            cv2.imshow('Frame', self.process_to_display_queue.popleft())
            cv2.waitKey(1)
    
    def cap_display(self):
        cap = cv2.VideoCapture(0)
        while True:
            _, frame_orig = cap.read()
            frame_orig = self.img_processor.horizontal_flip(frame_orig)
            self.cap_to_process_to_queue.append(frame_orig)
            if len(self.process_to_display_queue) == 0: continue
            cv2.imshow('Frame', self.process_to_display_queue.popleft())
            cv2.waitKey(1)
    
    def cap_display_independent(self):
        cap = cv2.VideoCapture(0)
        while True:
            _, frame_orig = cap.read()
            frame_orig = self.img_processor.horizontal_flip(frame_orig)
            self.cap_to_process_to_queue.append(frame_orig)
            # if len(self.process_to_display_queue) == 0: continue
            cv2.imshow('Frame', frame_orig)
            cv2.waitKey(1)
    



