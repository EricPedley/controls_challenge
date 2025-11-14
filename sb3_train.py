from env import TinyPhysicsEnv
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env

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
