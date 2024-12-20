import torch
from torch import hub

class ObjectDetector:
    def __init__(self, model_path):
        yolov = r"C:\yolov5-master"
        self.model = torch.hub.load(r'C:\yolov5-master', 'custom', path=model_path, source='local', force_reload=True)
        
        
    def get_model(self, choice="cpu"):
        if choice == "cuda" and torch.cuda.is_available() == True:
            device = torch.device("cuda")
            self.model.to(device)
        else:
            device = torch.device("cpu")
            self.model.to(device)
        return self.model
    