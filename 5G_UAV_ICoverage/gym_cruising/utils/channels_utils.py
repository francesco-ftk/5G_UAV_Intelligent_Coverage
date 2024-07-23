import numpy as np

from gym_cruising.geometry.point import Point
import math

UAV_ALTITUDE = 60  # 120 max altitude for law
a = 12.08  # in the dense urban case
b = 0.11  # in the dense urban case
nNLos = 23  # [dB] in the dense urban case
nLos = 1.6  # [dB] in the dense urban case
RATE_OF_GROWTH_G1 = -0.1
RATE_OF_GROWTH_G2 = 0.1
TRASMISSION_POWER = 30  # 30 dBm

LOS = []


# calculate distance between one UAV and one GU in air line
def calculate_distance_uav_gu(uav: Point, gu: Point):
    return math.sqrt(math.pow(uav.x_coordinate - gu.x_coordinate, 2) +
                     math.pow(uav.y_coordinate - gu.y_coordinate, 2) +
                     UAV_ALTITUDE ** 2)


# calculate the Probability of LoS link between one UAV and one GU
def get_PLoS(distance_uav_gu: float):
    elevation_angle = math.degrees(math.asin(UAV_ALTITUDE / distance_uav_gu))
    return 1 / (1 + a * math.exp((-1) * b * (elevation_angle - a)))


# return transition matrix for Markov Chain channel state update
def get_transition_matrix(distance_uav_gu: float, PLoS: float):
    PLoS2NLoS = 2 * ((1 - PLoS) / (1 + math.exp(RATE_OF_GROWTH_G1 * distance_uav_gu)) - (1 - PLoS) / 2)  # g1
    PNLoS2LoS = 2 * PLoS / (1 + math.exp(RATE_OF_GROWTH_G2 * distance_uav_gu))  # g2 # TODO check if correct
    return np.array([
        [1 - PLoS2NLoS, PLoS2NLoS],
        [PNLoS2LoS, 1 - PNLoS2LoS]
    ])


# calculate the Free Space PathLoss of the link between one UAV and one GU in dB
# 38.4684 is according to Friis equation with carrier frequency fc = 2GHz
def get_free_space_PathLoss(distance_uav_gu: float):
    return 20 * math.log(distance_uav_gu, 10) + 38.4684


# calculate PathLoss of the link between one UAV and one GU in dB
def get_PathLoss(distance_uav_gu: float, current_state: int):
    FSPL = get_free_space_PathLoss(distance_uav_gu)
    if current_state == 0:
        return FSPL + nLos
    return FSPL + nNLos


def getSINR(path_loss: float, interference_path_loss: float):
    return (path_loss + TRASMISSION_POWER) - (interference_path_loss - 111)

# check if the connection is failed
# def is_connection_failed(pl: float) -> bool:
#     return pl > 81.34641738844708

# def dB2W(decibel_value: float):
#     return 10 ** (decibel_value / 10)
#
#
# def dBm2W(dBm_value: float):
#     return math.pow(10, (dBm_value - 30) / 10)
#
#
# def W2dB(watt_value: float):
#     return math.log(watt_value, 10) * 10
