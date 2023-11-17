from ultralytics import YOLO
from ultralytics.utils.checks import check_amp
import intel_extension_for_pytorch as ipex
import torch

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
device_str = "xpu:0"
device = torch.device(device_str)

model = model.to(device)
check_amp(model)