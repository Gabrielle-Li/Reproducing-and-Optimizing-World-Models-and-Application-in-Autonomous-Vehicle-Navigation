import math
import sys
import time

import pyautogui
import pygame
from Cars import Car, Car_1  # 假设这是导入车辆类的语句

class Car_Environment:
    def __init__(self):
        super(Car_Environment, self).__init__()
        # 定义地图的大小和起始位置、目标位置

        self.size = width, height = (1500, 844)

        self.others = pygame.sprite.Group()
        self.player = Car(100, 85, 0)

        self.other_car_1 = Car_1(700, 700, 'resource/Car_3.png')  # 路上其他车或者障碍物
        self.other_car_2 = Car_1(450, 365, 'resource/Car_3.png')  # 路上其他车或者障碍物
        self.other_car_3 = Car_1(305, 650, 'resource/Car_4.png')  # 路上其他车或者障碍物
        self.other_car_4 = Car_1(1195, 380, 'resource/Car_1.png')  # 路上其他车或者障碍物

        self.others.add(self.other_car_1)
        self.others.add(self.other_car_2)
        self.others.add(self.other_car_3)
        self.others.add(self.other_car_4)

        self.E = 0
        self.R = 0

        pygame.init()
        pygame.display.set_caption("赛车游戏")
        self.screen = pygame.display.set_mode(self.size)
        self.FPS = 30
        self.clock = pygame.time.Clock()
        self.background = pygame.image.load("resource/map.png")

    def reset(self):
        self.player = Car(100, 85, 0)  # 重置车辆位置
        self.R = 0  # 奖励置0
        self.E = 0  # 能耗置0

    def play(self, action):
        screenshot_counter = 0  # 截图计数器，用于生成唯一的文件名
        move_occurred = False  # 初始化移动标志为 False

        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.player.image, self.player.rect)
        self.screen.blit(self.other_car_1.image, self.other_car_1.rect)
        self.screen.blit(self.other_car_2.image, self.other_car_2.rect)
        self.screen.blit(self.other_car_3.image, self.other_car_3.rect)
        self.screen.blit(self.other_car_4.image, self.other_car_4.rect)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 检查车辆是否移动
        if not move_occurred:
            old_rect = self.player.rect.copy()  # 备份原始位置
            self.simulate_key_press(action)
            self.player.move()  # 车辆移动
            if self.player.rect != old_rect:
                move_occurred = True  # 如果移动了，设置移动标志为 True

        if not self.is_within_lane(self.player):
            self.R = self.R - 99
            # print("超出车道！！！")

        if pygame.sprite.spritecollide(self.player, self.others, False):
            self.R = self.R - 99
            self.player.kill()
            # print("发生碰撞！！！")

        pygame.display.update()

        # 只有在车辆移动后才截图
        if move_occurred:
            self.E += 1
            self.R += 1

            # 先保存整个游戏画面的截图
            screenshot_name = f"screen/screenshot_{screenshot_counter}.png"
            pygame.image.save(self.screen, screenshot_name)
            # print(f"保存整个画面截图: {screenshot_name}")

            # 再保存车辆正前方小块区域的截图
            first_view_surface = self.capture_front_area(self.player)
            view_name = f"first_view/view_{screenshot_counter}.png"
            pygame.image.save(first_view_surface, view_name)
            # print(f"保存正前方小块区域截图: {view_name}")

            move_occurred = False  # 重置移动标志
            screenshot_counter += 1
        # move_occurred = False  # 重置移动标志

        self.clock.tick(self.FPS)

    def capture_front_area(self, player):
        # 定义截取区域的大小
        capture_width = 100
        capture_height = 100

        # 计算截取区域的左上角坐标
        capture_x = player.rect.centerx - capture_width // 2
        capture_y = player.rect.centery - capture_height // 2

        # 创建一个新的Surface对象，用于存储截取的图像
        captured_surface = pygame.Surface((capture_width, capture_height))

        # 截取游戏屏幕的指定区域到新的Surface对象
        captured_surface.blit(self.screen, (0, 0), pygame.Rect(capture_x, capture_y, capture_width, capture_height))

        return captured_surface

    def simulate_key_press(self, key):
        if key == 0:
            pyautogui.keyDown('up')
        elif key == 1:
            pyautogui.keyDown('down')
        elif key == 2:
            pyautogui.keyDown('left')
        elif key == 3:
            pyautogui.keyDown('right')

        time.sleep(0.5)  # 模拟按键操作后等待一段时间，可以根据需要调整
        # 松开所有按键
        pyautogui.keyUp('up')
        pyautogui.keyUp('down')
        pyautogui.keyUp('left')
        pyautogui.keyUp('right')

    def is_within_lane(self, player):
        # 根据背景图片，获取小车中心点位置的颜色，并判断是否在车道内
        background_color = self.background.get_at((player.rect.centerx, player.rect.centery))
        # 车道颜色为白色 RGB值
        lane_color = (191, 191, 191)
        return background_color == lane_color








