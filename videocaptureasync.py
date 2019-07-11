import threading
import cv2
import time
import logging


class VideoCaptureAsync:
    def __init__(self, src=0, width=480, height=300, skipframe=1):
        self.src = src
        self.grabbed = False
        tries = 0
        while not self.grabbed:
            print('Unable to connect to Stream..Retrying')
            self.cap = cv2.VideoCapture(self.src)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(0,cv2.CAP_PROP_BUFFERSIZE)
            self.grabbed, self.frame = self.cap.read()
            time.sleep(1)
            tries += 1
            if tries == 5:
               raise "Unable to connect to stream"
        self.started = False
        self.read_lock = threading.Lock()
        self.frame_count = 0
        self.skipframe = skipframe
        self.time_per_frame = 0
    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Starting asynchroneous video capturing')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            time_1 = time.time()
            grabbed = self.cap.grab()
            self.frame_count += 1
            if self.frame_count % self.skipframe  == 0:
                grabbed, frame = self.cap.retrieve()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            time_2 = time.time()
            self.time_per_frame = time_2-time_1
    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
