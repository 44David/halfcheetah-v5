import gymnasium as gym
import torch 
import torch.optim as optim
from continuous_ppo import ppo
from policy import Policy
from critic import Critic
import numpy as np
from running_mean_std import RunningMeanStd

def main():
    env = gym.make("HalfCheetah-v5")

    policy = Policy()
    critic = Critic()

    policy_optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)

    n_timesteps = 2048
    # observation space (-inf, inf) float64
    observation_space = 17
    # action space (-1, 1) float32
    action_space = 6

    cycles = 2000

    rms = RunningMeanStd(shape=env.observation_space.shape)

    raw_observation, info = env.reset()
    for i in range(cycles):
        
        print(f"Rollout epoch {i}")
        total_timesteps = 0
        
        trajectories = {
            "raw_observations": torch.zeros(n_timesteps, observation_space),
            "observations": torch.zeros(n_timesteps, observation_space),
            "actions": torch.zeros(n_timesteps, action_space),
            "log_probs": torch.zeros(n_timesteps),
            "rewards": torch.zeros(n_timesteps),
            "values": torch.zeros(n_timesteps),
            "episode_over": torch.zeros(n_timesteps)
        }    
        
        if i % 10 == 0:
            print(f"log std", policy.log_std.data)
        
        total_reward = 0
        for _ in range(n_timesteps):
            norm_observation = np.clip((raw_observation - rms.mean) / np.sqrt(rms.var + 1e-8), -10, 10)
            
            trajectories["raw_observations"][total_timesteps] =  torch.Tensor(raw_observation)
            trajectories["observations"][total_timesteps] = torch.Tensor(norm_observation)
            
            mean_vector = policy(torch.Tensor(norm_observation))
            value_estimate = critic(torch.Tensor(norm_observation))
            
            log_std = torch.clamp(policy.log_std, -20, -1.0)
            std = log_std.exp()
            
            dist = torch.distributions.Normal(mean_vector, std)
            pre_action = dist.rsample()
            action = torch.tanh(pre_action)
            
            log_prob = dist.log_prob(pre_action) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1)
            
            trajectories["actions"][total_timesteps] = action.detach()
            trajectories["log_probs"][total_timesteps] = log_prob.detach()
            trajectories["values"][total_timesteps] = value_estimate.detach()
            
            raw_observation, reward, terminated, truncated, info = env.step(action.detach().numpy())
            
            trajectories["rewards"][total_timesteps] =  reward
            trajectories["episode_over"][total_timesteps] = 1 if terminated or truncated else 0
            
            total_timesteps += 1
            total_reward += reward
            
            if terminated or truncated:
                print(f"*** total reward {total_reward}")
                total_reward = 0
                raw_observation, info = env.reset()
        
        rms.update(np.array(trajectories["raw_observations"]))
        policy, critic = ppo(trajectories, policy, critic, n_timesteps, policy_optimizer, critic_optimizer)            

    env.close()

    torch.save(policy.state_dict(), "halfcheetah_policy.pth")
    
    
    
if __name__ == "__main__":
    main()