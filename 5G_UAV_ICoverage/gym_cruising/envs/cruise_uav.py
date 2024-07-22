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
from gym_cruising.utils import channels_utils


class CruiseUAV(Cruise):
    uav = []
    gu = []
    pathLoss = []

    UAV_NUMBER = 3
    gu_number = 20
    UAV_RADIUS = 0.4
    MINIMUM_DISTANCE_BETWEEN_UAV = 1.5
    GU_RADIUS = 0.5

    SPAWN_GU_PROB = 0.005
    DISAPPEAR_GU_PROB = 0.001
    UAV_ALTITUDE = 120
    UAV_SENSING_ZONE_RADIUS = 50

    GU_MEAN_SPEED = 1.4  # 1.4 m/s
    GU_STANDARD_DEVIATION = 0.3  # 0.3 m/s  # va a 0 a circa 3 volte la deviazione standard

    def __init__(self,
                 render_mode=None, track_id: int = 1) -> None:
        super().__init__(render_mode, track_id)

        self.observation_space = spaces.Discrete(1)

        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        self.uav = []
        self.gu = []
        return super().reset(seed=seed, options=options)

    def perform_action(self, action: int) -> None:
        self.update_GU()

    def update_GU(self):
        # self.moveUAV()
        self.move_GU()
        # self.update_PathLoss_with_Markov_Chain()
        # self.check_if_disappear_GU()
        # self.check_if_spawn_new_GU()
        # self.check_connection_UAV_GU()

    # Random walk the GU
    def move_GU(self):
        area = self.np_random.choice(self.track.spawn_area)
        for gu in self.gu:
            repeat = True
            while repeat:
                previous_position = gu.position
                distance = self.np_random.normal(self.GU_MEAN_SPEED, self.GU_STANDARD_DEVIATION)
                direction = np.random.choice(['up', 'down', 'left', 'right'])

                if direction == 'up':
                    new_position = Point(previous_position.x_coordinate, previous_position.y_coordinate + distance)
                elif direction == 'down':
                    new_position = Point(previous_position.x_coordinate, previous_position.y_coordinate - distance)
                elif direction == 'left':
                    new_position = Point(previous_position.x_coordinate - distance, previous_position.y_coordinate)
                elif direction == 'right':
                    new_position = Point(previous_position.x_coordinate + distance, previous_position.y_coordinate)

                # TODO review if it's ok
                if new_position.is_in_area(area):
                    repeat = False
                    gu.position = new_position
                else:
                    repeat = True

    def update_PathLoss_with_Markov_Chain(self):
        for gu in self.gu:
            current_GU_PathLoss = []
            for index, uav in enumerate(self.uav):
                distance = channels_utils.calculate_distance_uav_gu(uav.position, gu.position)
                transition_matrix = channels_utils.get_transition_matrix(distance, gu.initial_PLoS[index])
                current_state = np.random.choice(range(len(transition_matrix)), p=transition_matrix[gu.initial_state[index]])
                path_loss = channels_utils.get_PathLoss(distance, current_state)
                current_GU_PathLoss.append(path_loss)
            self.pathLoss.append(current_GU_PathLoss)

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
                    path_loss = channels_utils.get_PathLoss(uav.position, gu.position)
                    if not channels_utils.is_connection_failed(path_loss):
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
            gu = GU(Point(x_coordinate, y_coordinate))
            for uav in self.uav:
                distance = channels_utils.calculate_distance_uav_gu(uav.position, gu.position)
                initial_PLoS = channels_utils.get_PLoS(distance)
                sample = random.random()
                if sample <= initial_PLoS:
                    gu.initial_state.append(0)  # 0 = LoS, 1 = NLoS
                else:
                    gu.initial_state.append(1)
                gu.initial_PLoS.append(initial_PLoS)
            self.gu.append(gu)

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

        # GU
        # for gu in self.gu:
        #     pygame.draw.circle(canvas,
        #                        gu.getColor(),
        #                        self.convert_point(gu.position),
        #                        self.GU_RADIUS * self.RESOLUTION)

        # GU image
        for gu in self.gu:
            canvas.blit(pygame.image.load(gu.getImage()), self.image_convert_point(gu.position))

        # UAV
        # for uav in self.uav:
        #     pygame.draw.circle(canvas,
        #                        Color.BLUE.value,
        #                        self.convert_point(uav.position),
        #                        self.UAV_RADIUS * self.RESOLUTION)

        # UAV image
        icon_drone = pygame.image.load('./gym_cruising/images/drone30.png')
        for uav in self.uav:
            canvas.blit(icon_drone, self.image_convert_point(uav.position))

    def convert_point(self, point: Point) -> Tuple[int, int]:
        pygame_x = (round(point.x_coordinate * self.RESOLUTION)
                    + self.X_OFFSET)
        pygame_y = (self.window_size
                    - round(point.y_coordinate * self.RESOLUTION)
                    + self.Y_OFFSET)
        return pygame_x, pygame_y

    def image_convert_point(self, point: Point) -> Tuple[int, int]:
        shiftX = 15
        shiftY = 15
        pygame_x = (round(point.x_coordinate * self.RESOLUTION) - shiftX + self.X_OFFSET)
        pygame_y = (self.window_size - round(point.y_coordinate * self.RESOLUTION) - shiftY + self.Y_OFFSET)
        return pygame_x, pygame_y

    def create_info(self) -> dict:
        return {"info": "sono presenti " + str(self.gu_number) + " GU"}
