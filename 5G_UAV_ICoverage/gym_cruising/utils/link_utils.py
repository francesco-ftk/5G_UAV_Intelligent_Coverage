from gym_cruising.geometry.point import Point
import math

UAV_ALTITUDE = 120
a = 12.08  # in the dense urban case
b = 0.11  # in the dense urban case
nNLos = 23  # dB
nLos = 1.6  # dB

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


# calculate the Free Space PathLoss of the link between one UAV and one GU in dB
# 38.4684 is according to Friis equation with carrier frequency fc = 2GHz
def get_free_space_PathLoss(distance_uav_gu: float):
    return 20 * math.log(distance_uav_gu, 10) + 38.4684


# calculate the spatial expectation of the PathLoss of the link between one UAV and one GU in dB
def get_PathLoss(uav: Point, gu: Point):
    distance_uav_gu = calculate_distance_uav_gu(uav, gu)
    PLoS = get_PLoS(distance_uav_gu)
    PNLoS = 1 - PLoS
    FSPL = get_free_space_PathLoss(distance_uav_gu)
    return 10 * math.log(
        math.pow(10, FSPL / 10) + math.pow(10, (PLoS * nLos) / 10) + math.pow(10, (PNLoS * nNLos) / 10), 10)


# check if the connection is failed
def is_connection_failed(pl: float) -> bool:
    return pl > 81.34641738844708


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