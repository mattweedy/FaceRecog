from threading import Thread
import cv2

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """
    
    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False
        
    def start(self):
        Thread(target=self.show, args=()).start()
        return self
    
    def show(self):
        while not self.stopped:
            if self.frame is not None:
                cv2.imshow('webcam', self.frame)
            if cv2.waitKey(1) == ord('q'):
                self.stopped = True
                cv2.destroyAllWindows()
                
    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()
        
    def __exit__(self, exc_type, exc_value, traceback):
        cv2.destroyAllWindows()