"""Core game loop and state machine."""

import os
import sys
import random
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
from rl.dual_role import DualRoleAlgorithm
from rendering.renderer import Renderer
from ui.menu import Menu
from ui.hud import HUD


def _get_model_path(algo_name: str) -> str:
    """Get standard model path for an algorithm."""
    safe_name = algo_name.lower().replace("-", "_").replace(" ", "_")
    return os.path.join(config.DEFAULT_MODEL_DIR, f"{safe_name}_model.pt")


class GameManager:
    def __init__(self, screen: pygame.Surface, clock: pygame.time.Clock):
        self.screen = screen
        self.clock = clock
        self.state = config.GameState.MENU
        self.mode = config.GameMode.PLAYER_MODE
        self.train_mode = config.TrainMode.TRAIN_LIVE
        self.selected_algorithm = "PPO"

        self.menu = Menu(screen)
        self.renderer = Renderer(screen)
        self.hud = HUD(screen)

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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

            result = self.menu.handle_event(event)
            if result:
                self.mode = result["mode"]
                self.selected_algorithm = result["algorithm"]
                self.train_mode = result["train_mode"]
                self._init_game()
                self.state = config.GameState.PLAYING

        self.menu.draw()
        pygame.display.flip()
        self.clock.tick(config.FPS)
        return True

    def _game_loop(self) -> bool:
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
        random.shuffle(spawn_points)
        entity_id = 0

        # Track shared inner algos for dual-role weight sharing
        shared_tagger = None
        shared_runner = None

        if self.mode == config.GameMode.PLAYER_MODE:
            if spawn_points:
                px, py = spawn_points.pop(0)
                self.player = Player(px, py, entity_id)
                self.entities.append(self.player)
                entity_id += 1

            for sp in spawn_points:
                algorithm = self._create_algorithm(algo_class,
                                                   shared_tagger, shared_runner)
                if shared_tagger is None and isinstance(algorithm, DualRoleAlgorithm):
                    shared_tagger = algorithm.tagger_algo
                    shared_runner = algorithm.runner_algo

                agent = Agent(sp[0], sp[1], entity_id, algorithm)
                self.agents.append(agent)
                self.entities.append(agent)
                entity_id += 1
        else:
            self.player = None
            for sp in spawn_points:
                algorithm = self._create_algorithm(algo_class,
                                                   shared_tagger, shared_runner)
                if shared_tagger is None and isinstance(algorithm, DualRoleAlgorithm):
                    shared_tagger = algorithm.tagger_algo
                    shared_runner = algorithm.runner_algo

                agent = Agent(sp[0], sp[1], entity_id, algorithm)
                self.agents.append(agent)
                self.entities.append(agent)
                entity_id += 1

        # Load trained model if using pre-trained mode
        if self.train_mode == config.TrainMode.USE_TRAINED and self.agents:
            model_path = _get_model_path(self.selected_algorithm)
            # DualRoleAlgorithm.load handles _tagger/_runner paths automatically
            self.agents[0].algorithm.load(model_path)

        # Spawn crates
        for cp in self.level.crate_spawns:
            self.movable_objects.append(MovableObject(cp[0], cp[1]))

        # Random tagger
        self.tag_logic = TagLogic(self.entities)
        if self.entities:
            tagger = random.choice(self.entities)
            self.tag_logic.set_tagger(tagger.entity_id)

        self.rl_env = TagEnvironment(self.level, self.entities,
                                     self.movable_objects)

    def _create_algorithm(self, algo_class, shared_tagger, shared_runner):
        """Create an algorithm instance, dual-role if enabled."""
        if config.DUAL_ROLE_ENABLED:
            return DualRoleAlgorithm(algo_class,
                                    shared_tagger=shared_tagger,
                                    shared_runner=shared_runner)
        return algo_class()

    def _update(self):
        """Update game state for one frame."""
        is_training = (self.train_mode == config.TrainMode.TRAIN_LIVE)

        keys = pygame.key.get_pressed()
        if self.player:
            self.player.handle_input(keys)

        for agent in self.agents:
            obs = self.rl_env.get_observation(agent)
            agent.decide_action(obs)

        for entity in self.entities:
            entity.apply_velocity()
            resolve_entity_walls(entity, self.level.wall_rects)

        for entity in self.entities:
            resolve_entity_crates(entity, self.movable_objects,
                                  self.level.wall_rects)

        tag_event = self.tag_logic.update()

        if tag_event:
            tagged_entity = None
            for e in self.entities:
                if e.entity_id == tag_event.get("tagged_id"):
                    tagged_entity = e
                    break
            if tagged_entity:
                self.renderer.emit_tag_particles(tagged_entity.x, tagged_entity.y)

        self.renderer.update_particles()

        if is_training:
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
        if self.player:
            self.renderer.set_camera(self.player.x, self.player.y)
        elif self.entities:
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
                      self.selected_algorithm, fps, self.train_mode)
