from gym_cruising.enums.color import Color
from gym_cruising.geometry.point import Point


class GU:
    position: Point
    connected: bool
    covered: bool
    transition_matrix = []

    def __init__(self, position: Point) -> None:
        self.position = position
        self.connected = False
        self.covered = False

    def getColor(self):
        if self.connected:
            if self.covered:
                return Color.GREEN.value
            else:
                return Color.YELLOW.value
        return Color.RED.value

    def getImage(self):
        if self.connected:
            if self.covered:
                return './gym_cruising/images/green30.png'
            else:
                return './gym_cruising/images/blue30.png'
        return './gym_cruising/images/white30.png'

    def setConnected(self, conneted: bool):
        self.connected = conneted
