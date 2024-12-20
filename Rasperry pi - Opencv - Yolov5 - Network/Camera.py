import cv2

class Cameras:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()