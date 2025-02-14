import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from tqdm import tqdm
import wandb

class GRPO:
    """Generalized Proximal Policy Optimization for training the robot brain."""
    
    def __init__(
        self,
        robot_brain,
        env,
        lr_llm=1e-5,
        lr_control=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        c1=1.0,
        c2=0.01,
        batch_size=64,
        n_epochs=10,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.robot_brain = robot_brain
        self.env = env
        self.device = device
        
        # Separate optimizers for LLM and control network
        self.optimizer_llm = torch.optim.AdamW(
            robot_brain.get_llm_parameters(),
            lr=lr_llm
        )
        self.optimizer_control = torch.optim.AdamW(
            robot_brain.get_control_parameters(),
            lr=lr_control
        )
        
        # Training parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Initialize logging with wandb
        wandb.init(project="robot-brain", config={
            "lr_llm": lr_llm,
            "lr_control": lr_control,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_epsilon": clip_epsilon,
            "c1": c1,
            "c2": c2,
            "batch_size": batch_size,
            "n_epochs": n_epochs
        })
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def collect_rollout(self, n_steps=1000):
        """Collect experience using current policy."""
        # Storage for rollout data
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        reasonings = []
        
        state, _ = self.env.reset()
        done = False
        
        for _ in range(n_steps):
            # Convert state to tensor
            vision = torch.FloatTensor(state['vision']).unsqueeze(0).to(self.device)
            robot_state = torch.FloatTensor(state['robot_state']).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                action, reasoning = self.robot_brain(
                    vision,
                    "Navigate to the nearest target while avoiding obstacles",
                    robot_state
                )
                action = action.cpu().numpy()[0]
            
            # Take action in environment
            next_state, reward, done, _, _ = self.env.step(action)
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            reasonings.append(reasoning)
            
            if done:
                state, _ = self.env.reset()
                done = False
            else:
                state = next_state
        
        return (
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(dones).to(self.device),
            reasonings
        )
    
    def train_iteration(self):
        """Run one iteration of GRPO training."""
        # Collect rollout data
        states, actions, rewards, dones, reasonings = self.collect_rollout()
        
        # Compute advantages and returns
        with torch.no_grad():
            values = self.robot_brain.get_value(states)
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        for _ in range(self.n_epochs):
            # Generate random permutation for minibatches
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get minibatch
                state_batch = states[batch_indices]
                action_batch = actions[batch_indices]
                advantage_batch = advantages[batch_indices]
                return_batch = returns[batch_indices]
                old_value_batch = values[batch_indices]
                
                # Forward pass
                new_actions, new_reasonings = self.robot_brain(
                    state_batch['vision'],
                    "Navigate to the nearest target while avoiding obstacles",
                    state_batch['robot_state']
                )
                
                # Compute losses
                # Policy loss (clipped objective)
                ratio = torch.exp(new_actions.log_prob(action_batch) - actions.log_prob(action_batch))
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_pred = self.robot_brain.get_value(state_batch)
                value_loss = F.mse_loss(value_pred, return_batch)
                
                # LLM loss (based on reasoning quality - simplified)
                llm_loss = F.mse_loss(
                    self.robot_brain.llm(new_reasonings).logits,
                    self.robot_brain.llm(reasonings).logits.detach()
                )
                
                # Combined loss
                total_loss = policy_loss + self.c1 * value_loss + self.c2 * llm_loss
                
                # Optimization step
                self.optimizer_llm.zero_grad()
                self.optimizer_control.zero_grad()
                total_loss.backward()
                self.optimizer_llm.step()
                self.optimizer_control.step()
                
                # Log metrics
                wandb.log({
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "llm_loss": llm_loss.item(),
                    "total_loss": total_loss.item()
                })
    
    def train(self, n_iterations=1000):
        """Train the robot brain for n iterations."""
        for i in tqdm(range(n_iterations)):
            self.train_iteration()
            
            # Save checkpoint every 100 iterations
            if (i + 1) % 100 == 0:
                torch.save({
                    'robot_brain_state_dict': self.robot_brain.state_dict(),
                    'optimizer_llm_state_dict': self.optimizer_llm.state_dict(),
                    'optimizer_control_state_dict': self.optimizer_control.state_dict(),
                }, f'checkpoint_{i+1}.pt')
        
        wandb.finish() 