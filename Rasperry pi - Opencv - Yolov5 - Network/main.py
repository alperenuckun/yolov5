from Camera import Cameras
from ObjectDetector import ObjectDetector
import cv2
import torch

def main():
    cam = Cameras(url='http://192.168.1.109:8080/')
    Object = ObjectDetector(model_path=r"C:\yolov5-master\best.pt")
    model = Object.get_model(choice="cuda")
    while True:
        frame = cam.read_frame()
        if frame is None:
            break
        result = model(frame, size=640)
        result.render()
        cv2.imshow("Main", result.ims[0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()