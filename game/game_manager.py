"""Core game loop and state machine."""

import sys
import importlib
import pygame
import config
from world.level import Level
from entities.player import Player
from entities.agent import Agent
from entities.movable_object import MovableObject
from game.tag_logic import TagLogic
from physics.collision import resolve_entity_walls, resolve_entity_crates
from rl.environment import TagEnvironment
from rendering.renderer import Renderer
from ui.menu import Menu
from ui.hud import HUD


class GameManager:
    def __init__(self, screen: pygame.Surface, clock: pygame.time.Clock):
        self.screen = screen
        self.clock = clock
        self.state = config.GameState.MENU
        self.mode = config.GameMode.PLAYER_MODE
        self.selected_algorithm = "Q-Learning"

        self.menu = Menu(screen)
        self.renderer = Renderer(screen)
        self.hud = HUD(screen)

        # Game objects (initialized when game starts)
        self.level = None
        self.entities = []
        self.agents = []
        self.player = None
        self.movable_objects = []
        self.tag_logic = None
        self.rl_env = None
        self.tick = 0

    def run(self):
        """Main loop."""
        running = True
        while running:
            if self.state == config.GameState.MENU:
                running = self._menu_loop()
            elif self.state == config.GameState.PLAYING:
                running = self._game_loop()
        return

    def _menu_loop(self) -> bool:
        """Handle menu. Returns False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

            result = self.menu.handle_event(event)
            if result:
                self.mode = result["mode"]
                self.selected_algorithm = result["algorithm"]
                self._init_game()
                self.state = config.GameState.PLAYING

        self.menu.draw()
        pygame.display.flip()
        self.clock.tick(config.FPS)
        return True

    def _game_loop(self) -> bool:
        """Handle one frame of gameplay. Returns False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.state = config.GameState.MENU
                return True

        self._update()
        self._render()
        pygame.display.flip()
        self.clock.tick(config.FPS)
        self.tick += 1
        return True

    def _init_game(self):
        """Set up a new game session."""
        self.tick = 0
        self.level = Level("level_01.txt")
        self.entities = []
        self.agents = []
        self.movable_objects = []

        # Load the selected RL algorithm class
        algo_module_path, algo_class_name = config.RL_ALGORITHMS[self.selected_algorithm]
        module = importlib.import_module(algo_module_path)
        algo_class = getattr(module, algo_class_name)

        spawn_points = list(self.level.spawn_points)
        entity_id = 0

        if self.mode == config.GameMode.PLAYER_MODE:
            # First spawn is the player
            if spawn_points:
                px, py = spawn_points.pop(0)
                self.player = Player(px, py, entity_id)
                self.entities.append(self.player)
                entity_id += 1

            # Remaining spawns are agents
            for sp in spawn_points:
                algorithm = algo_class()
                agent = Agent(sp[0], sp[1], entity_id, algorithm)
                self.agents.append(agent)
                self.entities.append(agent)
                entity_id += 1
        else:
            # Simulation mode: all agents
            self.player = None
            for sp in spawn_points:
                algorithm = algo_class()
                agent = Agent(sp[0], sp[1], entity_id, algorithm)
                self.agents.append(agent)
                self.entities.append(agent)
                entity_id += 1

        # Spawn crates
        for cp in self.level.crate_spawns:
            self.movable_objects.append(MovableObject(cp[0], cp[1]))

        # Initialize tag logic (first entity is tagger)
        self.tag_logic = TagLogic(self.entities)
        if self.entities:
            self.tag_logic.set_tagger(self.entities[0].entity_id)

        # Initialize RL environment
        self.rl_env = TagEnvironment(self.level, self.entities,
                                     self.movable_objects)

    def _update(self):
        """Update game state for one frame."""
        # 1. Gather actions
        keys = pygame.key.get_pressed()
        if self.player:
            self.player.handle_input(keys)

        for agent in self.agents:
            obs = self.rl_env.get_observation(agent)
            agent.decide_action(obs)

        # 2. Apply movement and resolve collisions
        for entity in self.entities:
            entity.apply_velocity()
            resolve_entity_walls(entity, self.level.wall_rects)

        for entity in self.entities:
            resolve_entity_crates(entity, self.movable_objects,
                                  self.level.wall_rects)

        # 3. Check tag
        tag_event = self.tag_logic.update()

        # 3b. Tag particles
        if tag_event:
            tagged_entity = None
            for e in self.entities:
                if e.entity_id == tag_event.get("tagged_id"):
                    tagged_entity = e
                    break
            if tagged_entity:
                self.renderer.emit_tag_particles(tagged_entity.x, tagged_entity.y)

        # 4. Update particles
        self.renderer.update_particles()

        # 5. RL feedback
        rewards = self.rl_env.get_all_rewards(tag_event)
        for agent in self.agents:
            next_obs = self.rl_env.get_observation(agent)
            reward = rewards.get(agent.entity_id, 0.0)
            done = (tag_event is not None and
                    tag_event.get("tagged_id") == agent.entity_id)
            agent.learn(reward, next_obs, done)

        if tag_event:
            for agent in self.agents:
                if agent.entity_id == tag_event.get("tagged_id"):
                    agent.algorithm.reset()

    def _render(self):
        """Render the game."""
        # Camera follows player in player mode, or first entity in sim mode
        if self.player:
            self.renderer.set_camera(self.player.x, self.player.y)
        elif self.entities:
            # In simulation, follow the tagger
            for e in self.entities:
                if e.is_tagger:
                    self.renderer.set_camera(e.x, e.y)
                    break

        lw, lh = self.level.get_pixel_dimensions()
        self.renderer.clamp_camera(lw, lh)

        self.renderer.draw_level(self.level)
        self.renderer.draw_movable_objects(self.movable_objects)
        self.renderer.draw_entities(self.entities,
                                    self.tag_logic.current_tagger_id,
                                    self.tick)

        self.renderer.draw_particles()

        fps = self.clock.get_fps()
        self.hud.draw(self.entities, self.tag_logic, self.mode,
                      self.selected_algorithm, fps)
