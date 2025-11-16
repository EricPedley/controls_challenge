from . import BaseController
import numpy as np
from skrl.utils.runner.torch import Runner
from skrl_train import make_env_and_agent
import torch


class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    _env, self.agent = make_env_and_agent(device='cpu')

    self.agent.load('/home/miller/code/controls_challenge/runs/25-11-15_18-26-33-859891_PPO/checkpoints/best_agent.pt')
    self.agent._random_timesteps=-1
    self.state_history = [(0,0,0)]*4
    self.target_lataccel_history = [0]*4
    self.current_lataccel_history = [0]*4
    self.action_history = [0]*4

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    with torch.no_grad():
        future_plan_array = np.zeros((4,49))
        future_plan_array[:, :len(future_plan[0])] = future_plan
        self.target_lataccel_history.append(target_lataccel)
        self.current_lataccel_history.append(current_lataccel)
        self.state_history.append(state)
        obs = np.concatenate([
            np.array(self.state_history[-4:]).flatten(),
            self.action_history[-4:],
            self.current_lataccel_history[-4:],
            self.target_lataccel_history[-4:],
            future_plan_array[:, 0:4].flatten()
        ])
        action = self.agent.act(torch.tensor(obs, dtype=torch.float32), 0, None)[0].item()
        self.action_history.append(action)
        return action
