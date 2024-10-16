import pyautogui
import time

def simulate_key_press(key):
    if key == 0:
        pyautogui.keyDown('up')
    elif key == 1:
        pyautogui.keyDown('down')
    elif key == 2:
        pyautogui.keyDown('left')
    elif key == 3:
        pyautogui.keyDown('right')

if __name__ == '__main__':
    while True:
        try:
            user_input = 2
            simulate_key_press(user_input)
            time.sleep(0.5)  # 模拟按键操作后等待一段时间，可以根据需要调整
            # 松开所有按键
            pyautogui.keyUp('up')
            pyautogui.keyUp('down')
            pyautogui.keyUp('left')
            pyautogui.keyUp('right')

        except ValueError:
            print("输入无效，请输入一个有效的数字。")
        except KeyboardInterrupt:
            print("\n程序终止")
            break
