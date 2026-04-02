"""Entry point for the Tag RL game."""

import pygame
from game.game_manager import GameManager
import config


def main():
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("TAG - RL Laboratory")
    clock = pygame.time.Clock()

    manager = GameManager(screen, clock)
    manager.run()

    pygame.quit()


if __name__ == "__main__":
    main()
