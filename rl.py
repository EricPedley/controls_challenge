from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTEXT_LENGTH, MAX_ACC_DELTA, CONTROL_START_IDX, STEER_RANGE, State, FuturePlan, COST_END_IDX, LAT_ACCEL_COST_MULTIPLIER , DEL_T
from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from controllers import BaseController
from controllers.pid import Controller as PIDController
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path


class TinyPhysicsEnv(Env, TinyPhysicsSimulator):
    def __init__(self, model_path='models/tinyphysics.onnx', data_path = 'data/00000.csv'):
        self.previous_pred = 0
        Env.__init__(self)
        TinyPhysicsSimulator.__init__(self, TinyPhysicsModel(model_path, False), data_path, BaseController())
        self.observation_space = Box(
            np.array([0, -1, -0.1, -2, -2]), 
            np.array([60, 1, 0.1, 2, 2])
        )
        self.action_space = Box(-2, 2, (1,))
    
    def step(self, action):
        state, target, futureplan = self.get_state_target_futureplan(self.step_idx)
        self.state_history.append(state)
        self.target_lataccel_history.append(target)
        self.futureplan = futureplan

        if self.step_idx < CONTROL_START_IDX:
            action = self.data['steer_command'].values[self.step_idx]
        else:
            action = action[0]
        action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        self.action_history.append(action)

        pred = self.sim_model.get_current_lataccel(
            sim_states=self.state_history[-CONTEXT_LENGTH:],
            actions=self.action_history[-CONTEXT_LENGTH:],
            past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
        if self.step_idx >= CONTROL_START_IDX:
            self.current_lataccel = pred
        else:
            self.current_lataccel = self.get_state_target_futureplan(self.step_idx)[1]

        self.current_lataccel_history.append(self.current_lataccel)


        self.step_idx += 1

        # obs, reward, terminated, truncated, info,  done
        # if self.step_idx == len(self.target_lataccel_history):
        #     obs = np.zeros(3)
        # else:
        obs = np.array(list(self.state_history[self.step_idx-1])+[target, self.current_lataccel])

        lat_accel_cost = np.mean((target - pred)**2)
        jerk_cost = np.mean(((pred-self.previous_pred) / DEL_T)**2)
        self.previous_pred = pred
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
        reward = -total_cost
        truncated = self.step_idx >= len(self.data)
        terminated = False
        done = truncated or terminated
        info = dict()
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None):
        TinyPhysicsSimulator.reset(self)
        return self.step(0)[0], dict()
    
if __name__ == "__main__":
    writer = SummaryWriter()

    experiment_logdir = f"runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    Path(experiment_logdir).mkdir(parents=True, exist_ok=True)


    def make_env():
        env = TinyPhysicsEnv()
        return env
    env = make_vec_env(make_env, n_envs=1)
    model = PPO("MlpPolicy", env, tensorboard_log=experiment_logdir, device='cpu')
        
    model.learn(total_timesteps=1_000_000)
    model.save(f'{experiment_logdir}/weights')
