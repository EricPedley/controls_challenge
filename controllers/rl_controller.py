from stable_baselines3 import PPO
from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.model = PPO.load("/home/miller/code/controls_challenge/runs/2025-11-12_15-26-19/weights.zip", device='cpu')
    self.state_history = [(0,0,0)]*20
    self.target_lataccel_history = [0]*20
    self.current_lataccel_history = [0]*20
    self.action_history = [0]*20

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    future_plan_array = np.zeros((4,49))
    future_plan_array[:, :len(future_plan[0])] = future_plan
    self.target_lataccel_history.append(target_lataccel)
    self.current_lataccel_history.append(current_lataccel)
    self.state_history.append(state)
    obs = np.concatenate([
        np.array(self.state_history[-20:]).flatten(),
        self.action_history[-20:],
        self.current_lataccel_history[-20:],
        self.target_lataccel_history[-20:],
        future_plan_array[:, 0:20].flatten()
    ])
    action = self.model.predict(obs)[0][0]
    self.action_history.append(action)
    return action
