from aimtools.config import *
from models.experimental import attempt_load
from utils.torch_utils import select_device


def get_model():
    # 选择设备
    device = select_device('')  # device 设备
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(WEIGHTS, map_location=device)

    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    return model, device, half, stride, names