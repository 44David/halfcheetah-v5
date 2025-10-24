import torch 

def ppo(trajectories, policy, critic, n_timesteps, policy_optimizer, vf_optimizer):
    
    observations = trajectories["observations"].detach()
    actions = trajectories["actions"].detach()
    log_prob = trajectories["log_probs"].detach()
    rewards = trajectories["rewards"].detach()
    value_estimates = trajectories["values"].detach()    
    episode_over = trajectories["episode_over"].detach()
    
    value_estimates = torch.cat((value_estimates, torch.Tensor([0.0])), dim=0)
    
    value_estimates = value_estimates.detach()
    advantage = compute_advantage(rewards, value_estimates, episode_over)
    advantage = advantage.detach()
    
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    
    value_targets = advantage + value_estimates[:-1]
    
    epochs = 10
    batch_size = 64
    EPS = 0.2
    
    for _ in range(epochs):
        
        shuffled = torch.randperm(n_timesteps)

        shuffled_observations = observations[shuffled]
        shuffled_actions = actions[shuffled]
        shuffled_log_prob = log_prob[shuffled]
        shuffled_advantages = advantage[shuffled].detach()
        shuffled_value_estimates = value_estimates[:-1][shuffled].detach()
        shuffled_value_targets = value_targets[shuffled]
        
        
        for i in range(0, n_timesteps, batch_size):
            
            batch_observations = shuffled_observations[i:i+batch_size]
            batch_actions = shuffled_actions[i:i+batch_size]
            batch_log_prob = shuffled_log_prob[i:i+batch_size]
            batch_advantages = shuffled_advantages[i:i+batch_size]
            batch_value_estimates = shuffled_value_estimates[i:i+batch_size]
            batch_value_targets = shuffled_value_targets[i:i+batch_size]
            
            mean_vector = policy(batch_observations)
            log_std = torch.clamp(policy.log_std, -20, -1.0)
            std = log_std.exp()
            
            dist = torch.distributions.Normal(mean_vector, std)
            entropy = dist.entropy().sum(dim=-1)
            mean_entropy = entropy.mean()
            
            pre_tanh = 0.5 * torch.log((1 + batch_actions + 1e-6) / (1 - batch_actions + 1e-6))
            
            new_log_probs = dist.log_prob(pre_tanh) - torch.log(1 - batch_actions.pow(2) + 1e-6)
            new_log_probs = new_log_probs.sum(dim=-1)
            
            
            vf = critic(batch_observations)
            
            probability_ratio = (new_log_probs - batch_log_prob).exp()

            clip_term = torch.clamp(probability_ratio, 1 - EPS, 1 + EPS)
            
            
            # negative since performing gradient ascent 
            clipped_surrogate_objective = -torch.mean(torch.min(probability_ratio * batch_advantages, clip_term * batch_advantages))
            
            clipped_vf = batch_value_estimates + torch.clamp(vf - batch_value_estimates, -EPS, +EPS)
            vf_loss_clipped = (clipped_vf - batch_value_targets)**2
            vf_loss_unclipped = (vf - batch_value_targets)**2
            vf_loss = torch.mean(torch.maximum(vf_loss_clipped, vf_loss_unclipped))
            
            
            c1 = 0.5
            c2 = 0.01
            
            total_loss = clipped_surrogate_objective + c1 * vf_loss - c2 * mean_entropy  
            
            policy_optimizer.zero_grad()
            vf_optimizer.zero_grad()
            
            
            total_loss.backward()
            
            policy_optimizer.step()
            vf_optimizer.step()
            
            
            

    return policy, critic

def compute_advantage(rewards, value_estimates, episode_over):
    gae = 0
    discount_factor = 0.99
    gae_lambda = 0.95
    advantages = torch.zeros_like(rewards)
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + discount_factor * value_estimates[t+1] * (1-episode_over[t]) - value_estimates[t]
        gae = delta + discount_factor * gae_lambda * (1 - episode_over[t]) * gae
        advantages[t] = gae
        
    return advantages
        
    