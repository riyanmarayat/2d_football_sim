"""
Main environment (inheriting gymnasium
"""

import functools

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

class FootballMAEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "football_v0"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.agents = [f"player{i}" for i in range(1, 23)]
        self.possible_agents = self.agents[:]

        self.action_spaces = {
            agent: spaces.Discrete(5) for agent in self.agents # up, down, left, right, kick
        }

        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(20,), dtype=np.float32)
            for agent in self.agents
        }

        self.state = None

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.state = {
            agent: np.random.rand(20).astype(np.float32) for agent in self.agents
        }
        observations = {agent: self.state[agent] for agent in self.agents}
        return observations

    def step(self, actions):
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.agents:
            self.state[agent] = np.random.rand(20).astype(np.float32)
            observations[agent] = self.state[agent]
            rewards[agent] = 0.0
            terminations[agent] = False
            truncations[agent] = False
            infos[agent] = {}

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode == "human":
            print("Rendering environment (not implemented)")

    def close(self):
        pass

def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    environment = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        environment = wrappers.CaptureStdoutWrapper(environment)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment

def raw_env(render_mode=None):
    environment = FootballMAEnv(render_mode=render_mode)
    return parallel_to_aec(environment)