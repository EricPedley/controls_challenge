from rl import TinyPhysicsEnv, BaseController
from controllers.pid import Controller as PIDController
from controllers.rl_controller import Controller as RLController
import numpy as np

def eval_reward_with_controller(env: TinyPhysicsEnv, controller: BaseController):
    done = False
    obs, _ = env.reset()
    target_lataccel = obs[(3+3)*env.policy_history_len-1]
    current_lataccel = obs[(2+3)*env.policy_history_len-1]
    state = obs[(3*env.policy_history_len-3):3*env.policy_history_len]
    future_plan = obs[6*env.policy_history_len:].reshape((4,-1))
    
    rewards = []
    while not done:
        action = [controller.update(target_lataccel, current_lataccel, state, future_plan)]
        obs, rew, term, truncated, _= env.step(action)
        done = term or truncated
        rewards.append(rew)
    return rewards

if __name__ == '__main__':

    env = TinyPhysicsEnv()
    pid_controller = PIDController()
    rl_controller = RLController()
    pid_rewards = eval_reward_with_controller(env, pid_controller)
    rl_rewards = eval_reward_with_controller(env, rl_controller)

    print("pid rewards", np.sum(pid_rewards))
    print("rl rewards", np.sum(rl_rewards))
