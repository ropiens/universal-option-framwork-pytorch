from gym.envs.registration import register

from asset.continuous_mountain_car import Continuous_MountainCarEnv
from asset.pendulum import PendulumEnv

register(
    id="MountainCarContinuous-h-v1",
    entry_point="asset:Continuous_MountainCarEnv",
    max_episode_steps=400,
)

register(
    id="Pendulum-h-v1",
    entry_point="asset:PendulumEnv",
)
