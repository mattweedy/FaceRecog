import cv2
import numpy as np
import face_recognition
from threading import Thread

class FrameProcessor:
    def __init__(self, video_grabber, video_shower, cps, compareFacesFn, skip_frames=5):
        self.grabber = video_grabber
        self.shower = video_shower
        self.cps = cps
        self.compareFaces = compareFacesFn
        self.stopped = False
        self.thread = None
        self.skip_frames = skip_frames
        self.frame_counter = 0

    def start(self):
        self.thread = Thread(target=self.process_frames, args=())
        self.thread.start()
        return self

    def process_frames(self):
        while not self.stopped:
            if self.grabber.stopped or self.shower.stopped:
                break
            
            frame = self.grabber.frame
            
            # process faces every skip_frames frames
            # for better performance
            if self.frame_counter % self.skip_frames == 0:
                imgSmall = cv2.resize(frame, (0,0), None, 0.25, 0.25)
                imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
                facesCurrFrame = face_recognition.face_locations(imgSmall)
                encodeCurrFrame = face_recognition.face_encodings(imgSmall, facesCurrFrame)
                self.compareFaces(encodeCurrFrame, facesCurrFrame, frame)
            
            self.cps.increment()
            self.shower.frame = frame
            self.frame_counter += 1

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join()