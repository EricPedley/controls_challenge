#!/usr/bin/env python3
"""
Example training script for the AltitudeControlEnv using skrl
"""

import torch
import torch.nn as nn
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from env import TinyPhysicsEnv


# Set random seed for reproducibility
set_seed(42)


hidden_size = 64
class Policy(GaussianMixin, Model):
    """Policy network for the PPO agent"""
    
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    """Value network for the PPO agent"""
    
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

def make_env_and_agent(device='cuda'):
    # Create environment
    env = TinyPhysicsEnv(num_envs=100, device=device)
    device = env.device
    
    print(f"Environment: AltitudeControlEnv")
    print(f"Device: {device}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create memory
    mem_size = 2048
    memory = RandomMemory(memory_size=mem_size, num_envs=env.num_envs, device=device)
    
    # Create modelnm
    models = {}
    policy_net = Policy(env.observation_space, env.action_space, device)
    value_net = Value(env.observation_space, env.action_space, device)
    models["policy"] = policy_net
    models["value"] = value_net
    
    # Initialize models
    policy_net.init_parameters(mean=0.0, std=0.1)
    value_net.init_parameters(mean=0.0, std=1)
    
    # Configure PPO agent
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = mem_size
    cfg["learning_epochs"] = 10
    cfg["learning_rate"] = 3e-4
    cfg["mini_batches"] = 64
    cfg["value_loss_scale"] = 0.5
    cfg["entropy_loss_scale"] = 1.0
    
    # Create agent
    agent = PPO(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)
    
    return env, agent

def train_agent():
    """Train a PPO agent on the altitude control environment"""
    
    env, agent = make_env_and_agent()
    
    # Configure trainer
    cfg_trainer = {"timesteps": 100_000_000//env.num_envs, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    
    # Start training
    print("\nStarting training...")
    trainer.train()
    
    print("\nTraining completed!")


def test_agent(env, agent, num_episodes=5):
    """Test the trained agent"""
    
    agent.set_mode("eval")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        while True:
            with torch.no_grad():
                actions = agent.act(obs, timestep=0, timesteps=0)[0]
            
            obs, reward, terminated, truncated, info = env.step(actions)
            episode_reward += reward.item()
            step_count += 1
            
            if terminated.any() or truncated.any():
                break
        
        print(f"  Episode {episode + 1} completed:")
        print(f"    Steps: {step_count}")
        print(f"    Total reward: {episode_reward:.2f}")


if __name__ == "__main__":
    train_agent()