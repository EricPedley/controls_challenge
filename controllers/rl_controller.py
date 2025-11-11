from stable_baselines3 import PPO
from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.model = PPO.load("/home/miller/code/controls_challenge/runs/2025-11-10_15-22-03/weights.zip", device='cpu')

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    return self.model.predict(np.array(list(state)))[0][0]
