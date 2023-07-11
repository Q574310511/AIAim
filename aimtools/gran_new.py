import cv2

from aimtools.grabscreen import grab_screen

screen_width = 2560  # 屏幕的宽
screen_height = 1440  # 屏幕的高
GAME_LEFT, GAME_TOP, GAME_WIDTH, GAME_HEIGHT = screen_width // 3, screen_height // 3, screen_width // 3, screen_height // 3  # 游戏内截图区域
RESIZE_WIN_WIDTH, RESIZE_WIN_HEIGHT = screen_width // 5, screen_height // 5  # 显示窗口大小
monitor = {
    'left': GAME_LEFT,
    'top': GAME_TOP,
    'width': GAME_WIDTH,
    'height': GAME_HEIGHT
}
window_name = 'test'

while True:
    img = grab_screen(tuple(monitor.values()))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, RESIZE_WIN_WIDTH, RESIZE_WIN_HEIGHT)
    cv2.imshow(window_name, img)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        cv2.destroyAllWindows()
        exit('ESC ... ')
