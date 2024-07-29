import time

import gymnasium as gym
import numpy as np
import math
import random

# env = gym.make('gym_cruising:Cruising-v0', render_mode='human', track_id=1)
env = gym.make('gym_cruising:Cruising-v0', render_mode='human', track_id=2)

env.action_space.seed(42)
state, info = env.reset(seed=int(time.perf_counter()))  # 42

for _ in range(100):
    observation, reward, terminated, truncated, info = env.step(0)
    print(f'observation={observation} info={info}')

    if terminated:
        observation, info = env.reset()

env.close()
