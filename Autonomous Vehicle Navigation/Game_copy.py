import math
import sys

import pyautogui
import pygame
from Cars import Car, Car_1  # 假设这是导入车辆类的语句

pygame.init()
size = width, height = (1500, 844)

pygame.display.set_caption("赛车游戏")
screen = pygame.display.set_mode(size)
FPS = 30
clock = pygame.time.Clock()

others = pygame.sprite.Group()

player = Car(100, 85, 0)

other_car_1 = Car_1(700, 700, 'resource/Car_3.png')  # 路上其他车或者障碍物
other_car_2 = Car_1(450, 365, 'resource/Car_3.png')  # 路上其他车或者障碍物
other_car_3 = Car_1(305, 650, 'resource/Car_4.png')  # 路上其他车或者障碍物
other_car_4 = Car_1(1195, 380, 'resource/Car_1.png')  # 路上其他车或者障碍物

others.add(other_car_1)
others.add(other_car_2)
others.add(other_car_3)
others.add(other_car_4)

background = pygame.image.load("resource/map.png")




# def reset():
#     player = Car(100, 85, 0)    #重置车辆位置
#     R = 0      #奖励置0
#     E = 0      #能耗置0


def play():
    screenshot_counter = 0   # 截图计数器，用于生成唯一的文件名
    move_occurred = False  # 初始化移动标志为 False

    while True:
        screen.blit(background, (0, 0))
        screen.blit(player.image, player.rect)
        screen.blit(other_car_1.image, other_car_1.rect)
        screen.blit(other_car_2.image, other_car_2.rect)
        screen.blit(other_car_3.image, other_car_3.rect)
        screen.blit(other_car_4.image, other_car_4.rect)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 检查车辆是否移动
        if not move_occurred:
            old_rect = player.rect.copy()  # 备份原始位置
            player.move()  # 车辆移动
            if player.rect != old_rect:
                move_occurred = True  # 如果移动了，设置移动标志为 True
            if not is_within_lane(player):
                print("超出车道！！！")

        if pygame.sprite.spritecollide(player, others, False):
            player.kill()
            print("发生碰撞！！！")

        pygame.display.update()

        # 只有在车辆移动后才截图
        if move_occurred:
            # 先保存整个游戏画面的截图
            screenshot_name = f"screen/screenshot_{screenshot_counter}.png"
            pygame.image.save(screen, screenshot_name)
            print(f"保存整个画面截图: {screenshot_name}")

            # 再保存车辆正前方小块区域的截图
            first_view_surface = capture_front_area(player)
            view_name = f"first_view/view_{screenshot_counter}.png"
            pygame.image.save(first_view_surface, view_name)
            print(f"保存正前方小块区域截图: {view_name}")

            move_occurred = False  # 重置移动标志
            screenshot_counter += 1
        # move_occurred = False  # 重置移动标志

        clock.tick(FPS)


def capture_front_area(player):
    # 定义截取区域的大小
    capture_width = 100
    capture_height = 100

    # 计算截取区域的左上角坐标
    capture_x = player.rect.centerx - capture_width // 2
    capture_y = player.rect.centery - capture_height // 2

    # 创建一个新的Surface对象，用于存储截取的图像
    captured_surface = pygame.Surface((capture_width, capture_height))

    # 截取游戏屏幕的指定区域到新的Surface对象
    captured_surface.blit(screen, (0, 0), pygame.Rect(capture_x, capture_y, capture_width, capture_height))

    return captured_surface

def is_within_lane(player):
    # 根据背景图片，获取小车中心点位置的颜色，并判断是否在车道内
    background_color = background.get_at((player.rect.centerx, player.rect.centery))
    # 车道颜色为白色 RGB值
    lane_color = (191, 191, 191)
    return background_color == lane_color


if __name__ == '__main__':
    play()
