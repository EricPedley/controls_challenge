from env import TinyPhysicsEnv
from controllers import BaseController
from controllers.pid import Controller as PIDController
from controllers.sb3 import Controller as SB3Controller
from controllers.skrl import Controller as SKRLController
import numpy as np

def eval_reward_with_controller(env: TinyPhysicsEnv, controller: BaseController):
    done = False
    obs, _ = env.reset()
    
    rewards = []
    while not done:
        target_lataccel = obs[0][(3+3)*env.policy_history_len-1]
        current_lataccel = obs[0][(2+3)*env.policy_history_len-1]
        state = obs[0][(3*env.policy_history_len-3):3*env.policy_history_len]
        future_plan = obs[0][6*env.policy_history_len:].reshape((4,-1))
        action = np.array([controller.update(target_lataccel, current_lataccel, state, future_plan)])
        obs, rew, term, truncated, _= env.step(action)
        done = term or truncated
        rewards.append(rew)
    return rewards

if __name__ == '__main__':

    env = TinyPhysicsEnv(device='cpu')
    pid_controller = PIDController()
    rl_controller = SKRLController()
    pid_rewards = eval_reward_with_controller(env, pid_controller)
    rl_rewards = eval_reward_with_controller(env, rl_controller)

    print("pid rewards", np.sum(pid_rewards))
    print("rl rewards", np.sum(rl_rewards))
