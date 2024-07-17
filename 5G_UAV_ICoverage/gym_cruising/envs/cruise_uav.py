""" This module contains the Cruising environment class """
import math
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
from gym_cruising.geometry.pose import Pose


class CruiseUAV(Cruise):
    uav = []
    gu = []

    UAV_NUMBER = 3
    gu_number = 20
    UAV_RADIUS = 0.4
    MINIMUM_DISTANCE_BETWEEN_UAV = 1.5
    GU_RADIUS = 0.5


    DISTANCE_STANDARD_DEVIATION = 0.02
    ANGLE_STANDARD_DEVIATION = 0.02

    def __init__(self,
                 render_mode=None, track_id: int = 1) -> None:
        super().__init__(render_mode, track_id)

        self.observation_space = spaces.Discrete(1)

        self.action_space = spaces.Discrete(2)

    def perform_action(self, action: int) -> None:
        self.perform_GU_action()

    def perform_GU_action(self):
        # TODO
        x = 1

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

    def create_info(self) -> dict:
        return {"info": ""}
