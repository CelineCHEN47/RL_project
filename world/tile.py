"""Tile type definitions for the game world."""

from enum import Enum


class TileType(Enum):
    FLOOR = 0
    WALL = 1
    SPAWN = 2
    CRATE_SPAWN = 3
