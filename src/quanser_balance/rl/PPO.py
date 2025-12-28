# import stable_baselnes3 as sb3
from stable_baselines3 import PPO

class CustomPPO(PPO):
    pass

# class CustomPPO(PPO):

#     def __init__(*args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def _setup_learn(self) -> None:
#         super()._setup_learn()
#         # Custom setup code can be added here