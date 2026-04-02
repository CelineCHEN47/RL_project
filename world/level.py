"""Level loader and map representation."""

import os
import pygame
from world.tile import TileType
import config


TILE_CHAR_MAP = {
    "#": TileType.WALL,
    ".": TileType.FLOOR,
    "S": TileType.SPAWN,
    "C": TileType.CRATE_SPAWN,
    "D": TileType.FLOOR,  # doorway, treated as floor
}


class Level:
    def __init__(self, map_file: str):
        self.grid: list[list[TileType]] = []
        self.width = 0  # in tiles
        self.height = 0
        self.wall_rects: list[pygame.Rect] = []
        self.spawn_points: list[tuple[int, int]] = []
        self.crate_spawns: list[tuple[int, int]] = []
        self._load(map_file)

    def _load(self, map_file: str):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "maps", map_file)
        with open(path, "r") as f:
            lines = [line.rstrip("\n") for line in f.readlines() if line.strip()]

        self.height = len(lines)
        self.width = max(len(line) for line in lines)

        for row_idx, line in enumerate(lines):
            row = []
            for col_idx, char in enumerate(line):
                tile = TILE_CHAR_MAP.get(char, TileType.FLOOR)
                row.append(tile)

                px = col_idx * config.TILE_SIZE
                py = row_idx * config.TILE_SIZE

                if tile == TileType.WALL:
                    self.wall_rects.append(
                        pygame.Rect(px, py, config.TILE_SIZE, config.TILE_SIZE)
                    )
                elif tile == TileType.SPAWN:
                    self.spawn_points.append((px + config.TILE_SIZE // 2,
                                              py + config.TILE_SIZE // 2))
                elif tile == TileType.CRATE_SPAWN:
                    self.crate_spawns.append((px, py))

            # Pad short rows
            while len(row) < self.width:
                row.append(TileType.FLOOR)
            self.grid.append(row)

    def get_tile(self, grid_x: int, grid_y: int) -> TileType:
        if 0 <= grid_y < self.height and 0 <= grid_x < self.width:
            return self.grid[grid_y][grid_x]
        return TileType.WALL  # out of bounds = wall

    def is_wall(self, grid_x: int, grid_y: int) -> bool:
        return self.get_tile(grid_x, grid_y) == TileType.WALL

    def pixel_to_grid(self, px: float, py: float) -> tuple[int, int]:
        return int(px // config.TILE_SIZE), int(py // config.TILE_SIZE)

    def grid_to_pixel(self, gx: int, gy: int) -> tuple[int, int]:
        return gx * config.TILE_SIZE, gy * config.TILE_SIZE

    def get_pixel_dimensions(self) -> tuple[int, int]:
        return self.width * config.TILE_SIZE, self.height * config.TILE_SIZE
