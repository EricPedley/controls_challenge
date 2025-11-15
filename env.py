from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTEXT_LENGTH, MAX_ACC_DELTA, CONTROL_START_IDX, STEER_RANGE, State, FuturePlan, COST_END_IDX, LAT_ACCEL_COST_MULTIPLIER , DEL_T, FUTURE_PLAN_STEPS
import numpy as np
from gymnasium.vector import VectorEnv
from gymnasium.spaces import Box
from controllers import BaseController
from pathlib import Path
import torch

class TinyPhysicsEnv(VectorEnv):
    def __init__(self, num_envs=1):
        self.sim_model = TinyPhysicsModel('models/tinyphysics.onnx')
        self.num_envs = num_envs
        self.previous_pred = 0
        self.policy_history_len = 4
        self.lookahead_len = 4
        state_low = [-1,0,-0.1]
        state_high = [1,60,0.1]
        actions_low = [-2]
        actions_high = [2]
        preds_low = [-2]
        preds_high = [2]
        self.all_data_paths = list(Path('data').glob('*.csv')) 
        self.observation_space = Box(
            np.concatenate([
                np.tile(state_low, self.policy_history_len),
                np.tile(actions_low, self.policy_history_len),
                np.tile(preds_low, self.policy_history_len),
                np.tile(preds_low, self.policy_history_len),
                np.repeat(preds_low+state_low, self.lookahead_len) # future plan
            ]),
            np.concatenate([
                np.tile(state_high, self.policy_history_len),
                np.tile(actions_high, self.policy_history_len),
                np.tile(preds_high, self.policy_history_len),
                np.tile(preds_high, self.policy_history_len),
                np.repeat(preds_high+state_high, self.lookahead_len) # future plan
            ]),
        )
        self.action_space = Box(-2, 2, (1,))
        self.data_dfs = [None]*self.num_envs
        self.step_indices = torch.zeros(self.num_envs)
        self.reset()
    
    def step(self, actions: torch.Tensor):
        assert actions.shape[0] == self.num_envs
        assert actions.shape[1] == 1

        state, target, futureplan = self.get_state_target_futureplan(self.step_idx)
        for env_idx in range(self.num_envs):
            entire_state = self.data_dfs[env_idx].iloc[self.step_indices[env_idx]]
            state = state['roll_lataccel'], state['v_ego'], state['a_ego']
            target = state['target_lataccel']
            future_plan = FuturePlan(
                lataccel=self.data['target_lataccel'].values[self.step_indices[env_idx] + 1:self.step_indices[env_idx] + FUTURE_PLAN_STEPS].tolist(),
                roll_lataccel=self.data['roll_lataccel'].values[self.step_indices[env_idx] + 1:self.step_indices[env_idx] + FUTURE_PLAN_STEPS].tolist(),
                v_ego=self.data['v_ego'].values[self.step_indices[env_idx] + 1:self.step_indices[env_idx] + FUTURE_PLAN_STEPS].tolist(),
                a_ego=self.data['a_ego'].values[self.step_indices[env_idx] + 1:self.step_indices[env_idx] + FUTURE_PLAN_STEPS].tolist()
            )
        



        self.state_history.append(state)
        self.target_lataccel_history.append(target)

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
        # future_lat_accel = futureplan[0][0] if len(futureplan[0])>0 else self.current_lataccel
        future_plan_array = np.zeros((4,49))
        future_plan_array[:, :len(futureplan[0])] = futureplan
        obs = np.concatenate([
            np.array(self.state_history[-self.policy_history_len:]).flatten(),
            self.action_history[-self.policy_history_len:],
            self.current_lataccel_history[-self.policy_history_len:],
            self.target_lataccel_history[-self.policy_history_len:],
            future_plan_array[:, 0:self.lookahead_len].flatten()
        ])

        lat_accel_cost = (target - pred)**2
        jerk_cost = ((pred-self.previous_pred) / DEL_T)**2
        self.previous_pred = pred
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
        if self.step_idx < CONTROL_START_IDX:
            reward = 0
        else:
            reward = -total_cost
        truncated = self.step_idx >= len(self.data)
        terminated = False
        info = dict()
        return obs, reward, terminated, truncated, info
    
    def _reset_index(self, env_index: int):
        assert env_index < self.num_envs
        data_path = np.random.choice(self.all_data_paths)
        self.data_dfs[env_index] = TinyPhysicsSimulator.get_data(None, path)
    
    def reset(self, seed=None):
        self.env_indices = np.zeros(self.num_envs, dtype=np.uint32)




