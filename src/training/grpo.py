import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from tqdm import tqdm
import wandb

class GRPO:
    """Simplified Proximal Policy Optimization for training the robot control network."""
    
    def __init__(
        self,
        robot_brain,
        env,
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        batch_size=32,
        n_epochs=5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_wandb=False
    ):
        self.robot_brain = robot_brain
        self.env = env
        self.device = device
        self.use_wandb = use_wandb
        
        # Single optimizer for control network
        self.optimizer = torch.optim.Adam(
            robot_brain.get_parameters(),
            lr=lr
        )
        
        # Training parameters
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Initialize logging with wandb if enabled
        if self.use_wandb:
            wandb.init(project="robot-brain", config={
                "lr": lr,
                "gamma": gamma,
                "clip_epsilon": clip_epsilon,
                "batch_size": batch_size,
                "n_epochs": n_epochs
            })
    
    def collect_rollout(self, n_steps=100):  # Reduced from 1000 to 100 steps
        """Collect experience using current policy."""
        # Storage for rollout data
        vision_states = []
        robot_states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        
        state, _ = self.env.reset()
        done = False
        
        for _ in range(n_steps):
            # Convert state to tensor and fix vision shape from [H, W, C] to [C, H, W]
            vision = torch.FloatTensor(state['vision']).permute(2, 0, 1).unsqueeze(0).to(self.device)
            robot_state = torch.FloatTensor(state['robot_state']).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                dist, action, _ = self.robot_brain(
                    vision,
                    "Navigate to the nearest target while avoiding obstacles",
                    robot_state
                )
                log_prob = dist.log_prob(action).sum(-1)  # Sum log probs across action dimensions
                action = action.cpu().numpy()[0]
            
            # Take action in environment
            next_state, reward, done, _, _ = self.env.step(action)
            
            # Store experience
            vision_states.append(state['vision'])
            robot_states.append(state['robot_state'])
            actions.append(action)
            log_probs.append(log_prob.cpu().numpy())
            rewards.append(reward)
            dones.append(done)
            
            if done:
                state, _ = self.env.reset()
                done = False
            else:
                state = next_state
        
        # Convert to tensors
        vision_states = torch.FloatTensor(vision_states).permute(0, 3, 1, 2).to(self.device)  # [B, C, H, W]
        robot_states = torch.FloatTensor(robot_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        log_probs = torch.FloatTensor(log_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute returns
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return
        
        # Create state dictionary
        states = {
            'vision': vision_states,
            'robot_state': robot_states
        }
        
        return states, actions, log_probs, returns
    
    def train_iteration(self):
        """Run one iteration of PPO training."""
        # Collect rollout data
        states, actions, old_log_probs, returns = self.collect_rollout()
        
        # Training loop
        for _ in range(self.n_epochs):
            # Generate random permutation for minibatches
            indices = torch.randperm(len(states['vision']))
            
            for start in range(0, len(states['vision']), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get minibatch
                state_batch = {
                    'vision': states['vision'][batch_indices],
                    'robot_state': states['robot_state'][batch_indices]
                }
                action_batch = actions[batch_indices]
                old_log_prob_batch = old_log_probs[batch_indices]
                return_batch = returns[batch_indices]
                
                # Forward pass
                dist, _, _ = self.robot_brain(
                    state_batch['vision'],
                    "Navigate to the nearest target while avoiding obstacles",
                    state_batch['robot_state']
                )
                
                # Compute policy loss
                log_prob = dist.log_prob(action_batch).sum(-1)
                ratio = torch.exp(log_prob - old_log_prob_batch)
                surr1 = ratio * return_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * return_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Optimization step
                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()
                
                # Log metrics if wandb is enabled
                if self.use_wandb:
                    wandb.log({
                        "policy_loss": policy_loss.item(),
                        "mean_return": return_batch.mean().item()
                    })
    
    def train(self, n_iterations=100):  # Reduced from 1000 to 100 iterations
        """Train the robot brain for n iterations."""
        for i in tqdm(range(n_iterations)):
            self.train_iteration()
            
            # Save checkpoint every 10 iterations
            if (i + 1) % 10 == 0:  # Changed from 100 to 10
                torch.save({
                    'robot_brain_state_dict': self.robot_brain.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, f'checkpoint_{i+1}.pt')
        
        if self.use_wandb:
            wandb.finish() 