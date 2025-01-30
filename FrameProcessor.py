import cv2
import numpy as np
import face_recognition
from threading import Thread

class FrameProcessor:
    # def __init__(self, video_grabber, video_shower, compareFacesFn, skip_frames=5):
    def __init__(self, video_grabber, video_shower, compareFacesFn):
        self.grabber = video_grabber
        self.shower = video_shower
        self.compareFaces = compareFacesFn
        self.stopped = False
        self.thread = None
        # self.skip_frames = skip_frames
        # self.frame_counter = 0

    def start(self):
        self.thread = Thread(target=self.process_frames, args=())
        self.thread.start()
        return self

    def process_frames(self):
        while not self.stopped:
            if self.grabber.stopped or self.shower.stopped:
                break
        
            # grab the latest frame
            frame = self.grabber.frame.copy() # avoid race conditions
            
            # if self.frame_counter % self.skip_frames != 0:
            #     self.frame_counter += 1
            #     continue
            
            # downscale and use HOG for detection
            imgSmall = cv2.resize(frame, (0,0), fx=0.25, fy=0.25) # 1/4 size
            imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
            
            # use only face detection (no encoding) for tracking
            facesCurrFrame = face_recognition.face_locations(imgSmall, model="hog")
            
            # only encode faces if motion/new face is detected
            if len(facesCurrFrame) > 0:
                encodeCurrFrame = face_recognition.face_encodings(imgSmall, facesCurrFrame, num_jitters=1)
                self.compareFaces(encodeCurrFrame, facesCurrFrame, frame)

            # update the display frame
            self.shower.frame = frame
            # self.frame_counter += 1

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join()