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

    # RESOLUTION = 3
    TRACK1 = (1,
              (
                  Line(Point(0, 0), Point(0, 300)),
                  Line(Point(0, 300), Point(300, 300)),
                  Line(Point(300, 300), Point(300, 0)),
                  Line(Point(300, 0), Point(0, 0))
              ),
              (
                  ((10, 290), (10, 290)),
              )
              )

    # RESOLUTION = 0.25
    TRACK3 = (2,
              (
                  Line(Point(0, 0), Point(0, 4000)),
                  Line(Point(0, 4000), Point(4000, 4000)),
                  Line(Point(4000, 4000), Point(4000, 0)),
                  Line(Point(4000, 0), Point(0, 0))
              ),
              (
                  ((60, 3940), (60, 3940)),
              )
              )

    # numero, linee muri e area dove oggetti possono apparire (spawnare)
