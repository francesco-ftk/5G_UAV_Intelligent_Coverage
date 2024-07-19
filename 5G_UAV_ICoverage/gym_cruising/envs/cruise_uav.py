""" This module contains the Cruising environment class """
import math
import random
from typing import Optional, Tuple

import numpy as np
import pygame
from gymnasium.vector.utils import spaces
from pygame import Surface

from gym_cruising.actors.GU import GU
from gym_cruising.actors.UAV import UAV
from gym_cruising.enums.color import Color
from gym_cruising.envs.cruise import Cruise
from gym_cruising.geometry.point import Point
from gym_cruising.utils import link_utils


class CruiseUAV(Cruise):
    uav = []
    gu = []

    UAV_NUMBER = 3
    gu_number = 20
    UAV_RADIUS = 0.4
    MINIMUM_DISTANCE_BETWEEN_UAV = 1.5
    GU_RADIUS = 0.5

    SPAWN_GU_PROB = 0.005
    DISAPPEAR_GU_PROB = 0.001
    UAV_ALTITUDE = 120
    UAV_SENSING_ZONE_RADIUS = 50

    GU_STANDARD_DEVIATION = 2  # 4,6 per andare a 0 a circa 3 volte la deviazione standard -> 13,8 m/s

    def __init__(self,
                 render_mode=None, track_id: int = 1) -> None:
        super().__init__(render_mode, track_id)

        self.observation_space = spaces.Discrete(1)

        self.action_space = spaces.Discrete(2)

    def perform_action(self, action: int) -> None:
        self.update_GU()

    def update_GU(self):
        # self.moveUAV()
        self.move_GU()
        # self.check_if_disappear_GU()
        # self.check_if_spawn_new_GU()
        self.check_connection_UAV_GU()

    def move_GU(self):
        area = self.np_random.choice(self.track.spawn_area)
        for gu in self.gu:
            repeat = True
            while repeat:
                previous_position = gu.position
                x_noise = self.np_random.normal(0, self.GU_STANDARD_DEVIATION)
                y_noise = self.np_random.normal(0, self.GU_STANDARD_DEVIATION)
                new_position = Point(previous_position.x_coordinate + x_noise, previous_position.y_coordinate + y_noise)
                if new_position.is_in_area(area):
                    repeat = False
                    gu.position = new_position
                else:
                    repeat = True

    def check_if_disappear_GU(self):
        disappeared_GU = 0
        index_to_remove = []
        for i in range(self.gu_number):
            sample = random.random()
            if sample <= self.DISAPPEAR_GU_PROB:
                index_to_remove.append(i)
                disappeared_GU += 1
        index_to_remove = sorted(index_to_remove, reverse=True)
        for index in index_to_remove:
            del self.gu[index]
        self.gu_number -= disappeared_GU


    def check_if_spawn_new_GU(self):
        sample = random.random()
        for _ in range(4):
            if sample <= self.SPAWN_GU_PROB:
                area = self.np_random.choice(self.track.spawn_area)
                x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
                y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
                self.gu.append(GU(Point(x_coordinate, y_coordinate)))
                self.gu_number += 1

    def check_connection_UAV_GU(self):
        for gu in self.gu:
            gu.setConnected(False)
        for uav in self.uav:
            for gu in self.gu:
                if not gu.connected:
                    path_loss = link_utils.get_PathLoss(uav.position, gu.position)
                    if not link_utils.is_connection_failed(path_loss):
                        gu.setConnected(True)


    def get_observation(self) -> int:
        return 0

    def check_if_terminated(self) -> bool:
        return False

    def check_if_truncated(self) -> bool:
        return False

    def calculate_reward(self) -> float:
        reward = 0.0
        return reward

    def init_environment(self, options: Optional[dict] = None) -> None:
        self.init_uav()
        self.init_gu()

    def init_uav(self) -> None:
        area = self.np_random.choice(self.track.spawn_area)
        x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
        y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
        self.uav.append(UAV(Point(x_coordinate, y_coordinate)))
        for i in range(1, self.UAV_NUMBER):
            x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
            y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
            position = Point(x_coordinate, y_coordinate)
            while self.collision_avoided(i, position):
                x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
                y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
                position = Point(x_coordinate, y_coordinate)
            self.uav.append(UAV(position))

    def collision_avoided(self, uav_index, position):
        collision = False
        for j in range(uav_index):
            if self.uav[j].position.calculate_distance(position) <= self.MINIMUM_DISTANCE_BETWEEN_UAV:
                collision = True
                break
        return collision

    def init_gu(self) -> None:
        area = self.np_random.choice(self.track.spawn_area)
        for _ in range(self.gu_number):
            x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
            y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
            self.gu.append(GU(Point(x_coordinate, y_coordinate)))

    def draw(self, canvas: Surface) -> None:
        # CANVAS
        canvas.fill(Color.WHITE.value)

        # WALL
        for wall in self.world:
            pygame.draw.line(canvas,
                             Color.BLACK.value,
                             self.convert_point(wall.start),
                             self.convert_point(wall.end),
                             self.WIDTH)

        # UAV
        for uav in self.uav:
            pygame.draw.circle(canvas,
                               Color.BLUE.value,
                               self.convert_point(uav.position),
                               self.UAV_RADIUS * self.RESOLUTION)

        # UAV image
        icon_drone = pygame.image.load('./gym_cruising/images/drone1.png')
        for uav in self.uav:
            canvas.blit(icon_drone, self.drone_convert_point(uav.position))

        # GU
        for gu in self.gu:
            pygame.draw.circle(canvas,
                               gu.getColor(),
                               self.convert_point(gu.position),
                               self.GU_RADIUS * self.RESOLUTION)

    def convert_point(self, point: Point) -> Tuple[int, int]:
        pygame_x = (round(point.x_coordinate * self.RESOLUTION)
                    + self.X_OFFSET)
        pygame_y = (self.window_size
                    - round(point.y_coordinate * self.RESOLUTION)
                    + self.Y_OFFSET)
        return pygame_x, pygame_y

    def drone_convert_point(self, point: Point) -> Tuple[int,int]:
        shiftX = 25
        shiftY = 25
        pygame_x = (round(point.x_coordinate * self.RESOLUTION) - shiftX + self.X_OFFSET)
        pygame_y = (self.window_size - round(point.y_coordinate * self.RESOLUTION) - shiftY + self.Y_OFFSET)
        return pygame_x, pygame_y

    def create_info(self) -> dict:
        return {"info": "sono presenti " + str(self.gu_number) + " GU"}
