import mss
import numpy as np
import cv2
import torch
import win32con
import win32gui
import pynput
from aimtools.config import *
from aimtools.mouse import move_mouse
from load_model import get_model
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import Annotator, colors

sct = mss.mss()
screen_width = 2560  # 屏幕的宽
screen_height = 1440  # 屏幕的高
GAME_LEFT, GAME_TOP, GAME_WIDTH, GAME_HEIGHT = screen_width // 3, screen_height // 3, screen_width // 3, screen_height // 3  # 游戏内截图区域
RESIZE_WIN_WIDTH, RESIZE_WIN_HEIGHT = screen_width // 5, screen_height // 5  # 显示窗口大小
# monitor = {
#     'left': GAME_LEFT,
#     'top': GAME_TOP,
#     'width': GAME_WIDTH,
#     'height': GAME_HEIGHT
# }
monitor = {
    'left': 810,
    'top': 390,
    'width': 300,
    'height': 300
}
window_name = 'test'
# 获取模型、步幅以及类别
model, device, half, stride, names = get_model()
imgsz = check_img_size(IMGSZ, s=stride)  # check image size

# 实例化一个鼠标的控制对象
mouse_controller = pynput.mouse.Controller()


# 图像识别
# 不进行梯度处理
@torch.no_grad()
def pred_img(img0):
    # Padded resize
    img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    # 归一化处理
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0

    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    pred = model(img, augment=False, visualize=False)[0]
    # NMS
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, None, False, max_det=1000)
    # Process predictions
    det = pred[0]

    im0 = img0.copy()
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

    annotator = Annotator(im0, line_width=LINE_THICKNESS, example=str(names))
    xywh_list = []
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            xywh_list.append(xywh)
            label = None if HIDE_LABELS else (names[c] if HIDE_CONF else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))
    im0 = annotator.result()
    return im0, xywh_list


# 鼠标驱动
def mouse_lock_aim(xywhs, mouse, left, top, width, height):
    mouse_x, mouse_y = mouse.position
    dist_tuple = None
    for index, xywh in enumerate(xywhs):
        x, y, _, _ = xywh
        dist = ((x - mouse_x) ** 2 + (y - mouse_y) ** 2) ** .5
        if dist_tuple:
            _, old_dist, _ = dist_tuple
            if dist < old_dist:
                dist_tuple = (index, dist, xywh)
        else:
            dist_tuple = (index, dist, xywh)
    if dist_tuple:
        _, _, xywh = dist_tuple
        for i, v in enumerate(xywh):
            if i % 2 == 0:
                xywh[i] = v * width
            else:
                xywh[i] = v * height
        x, y, w, h = xywh

        move_mouse(x + left - mouse_x, y + top - mouse_y)


while True:
    img = sct.grab(monitor=monitor)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 将图片通道类型学转为BGR
    img, aims = pred_img(img)
    # mouse_lock_aim(aims, mouse_controller, **monitor)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL 根据窗口大小设置我们的图片大小
    cv2.resizeWindow(window_name, RESIZE_WIN_WIDTH, RESIZE_WIN_HEIGHT)
    cv2.imshow(window_name, img)
    k = cv2.waitKey(1)
    hwnd = win32gui.FindWindow(None, window_name)
    # 窗口需要正常大小且在后台，不能最小化
    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                          win32con.SWP_NOMOVE | win32con.SWP_NOACTIVATE | win32con.SWP_NOOWNERZORDER | win32con.SWP_SHOWWINDOW | win32con.SWP_NOSIZE)
    if k % 256 == 27:  # ESC
        cv2.destroyAllWindows()
        exit('ESC ...')
