""" This module contains the Track enum """
from enum import Enum
from typing import Tuple

from gym_cruising.geometry.line import Line
from gym_cruising.geometry.point import Point


class Track(Enum):
    # pylint: disable=invalid-name
    walls: Tuple[Line, ...]
    spawn_area: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]
    # pylint: enable=invalid-name

    def __new__(cls,
                value: int,
                walls: Tuple[Line, ...] = (),
                spawn_area: Tuple[Tuple[Tuple[float, float],
                                        Tuple[float, float]], ...] = ()):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.walls = walls
        obj.spawn_area = spawn_area
        return obj

    TRACK1 = (1,
              (
                  Line(Point(0, 0), Point(0, 90)),
                  Line(Point(0, 90), Point(90, 90)),
                  Line(Point(90, 90), Point(90, 0)),
                  Line(Point(90, 0), Point(0, 0))
              ),
              (
                  ((2, 88), (2, 88)),
              )
              )

    # numero, linee muri e area dove oggetti possono apparire (spawnare)
