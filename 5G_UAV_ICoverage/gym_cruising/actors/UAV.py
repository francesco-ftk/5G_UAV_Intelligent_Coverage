from gym_cruising.geometry.point import Point


class UAV:
    position: Point
    previous_position: Point

    def __init__(self, position: Point) -> None:
        self.position = position
        self.previous_position = position
