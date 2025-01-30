import cv2
from threading import Thread

# implementing multithreading from : https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/
class VideoGrab:
    """
    Class that continuously grabs frames from a VideoCapture object
    with a dedicated thread.
    """
    
    def __init__(self, src=0):
        # initliase video capture object and read first frame
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # var used to indicate if thread should be stopped
        self.stopped = False
        
    def start(self):
        # create thread to read frames
        Thread(target=self.get, args=()).start()
        return self
    
    def get(self):
        # continuously read frames from stream
        # stores in class's instance frame var
        # as long as thread is not stopped
        while not self.stopped:
            if not self.grabbed:
                # if webcam/video isn't grabbed, (turned off/video ends)
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                
    def stop(self):
        self.stopped = True
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()