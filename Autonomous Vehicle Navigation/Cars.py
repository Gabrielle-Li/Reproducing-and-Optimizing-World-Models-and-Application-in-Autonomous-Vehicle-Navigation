import pygame
import math


class Car(pygame.sprite.Sprite):
    def __init__(self, x, y, angle):
        super(Car, self).__init__()

        self.angle = angle
        self.original_image = pygame.image.load("resource/Car_2.png")  # 原始未旋转的图片
        self.image = self.original_image.copy()  # 复制一份作为当前显示的图片
        self.rect = self.image.get_rect(center=(x, y))
        self.width = self.image.get_width()
        self.height = self.image.get_height()

    def rotate(self, angle):
        # 旋转图片，并保持中心不变
        self.angle -= angle
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

    def move(self):
        press_keys = pygame.key.get_pressed()
        if press_keys[pygame.K_UP]:
            # 计算相对于当前角度的前进方向
            angle_radians = math.radians(self.angle)
            dx = 5 * math.cos(angle_radians)
            dy = 5 * math.sin(angle_radians)
            self.rect.move_ip(dx, -dy)  # 注意pygame的y轴向下为正方向
        if press_keys[pygame.K_DOWN]:
            # 后退相对于当前角度的方向
            angle_radians = math.radians(self.angle)
            dx = -5 * math.cos(angle_radians)
            dy = -5 * math.sin(angle_radians)
            self.rect.move_ip(dx, -dy)
        if press_keys[pygame.K_LEFT]:
            self.rotate(-1)
            self.rect.move_ip(0, 0)
        if press_keys[pygame.K_RIGHT]:
            self.rotate(1)
            self.rect.move_ip(0, 0)


class Car_1(pygame.sprite.Sprite):
    def __init__(self, x, y, path):
        super(Car_1, self).__init__()

        self.image = pygame.image.load(path)
        self.rect = self.image.get_rect(center=(x, y))

    def move(self):
        press_keys = pygame.key.get_pressed()
        if press_keys[pygame.K_UP]:
            self.rect.move_ip(0, -5)
        if press_keys[pygame.K_DOWN]:
            self.rect.move_ip(0, 5)
        if press_keys[pygame.K_LEFT]:
            self.rect.move_ip(-5, 0)
        if press_keys[pygame.K_RIGHT]:
            self.rect.move_ip(5, 0)
