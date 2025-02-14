import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from torch.distributions import Normal

class RobotControlNetwork(nn.Module):
    """Neural network for converting high-level commands to robot actions."""
    
    def __init__(self, vision_channels=3, hidden_size=256):
        super().__init__()
        
        # Vision processing
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(vision_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_size)
        )
        
        # Command processing
        self.command_encoder = nn.Linear(768, hidden_size)  # 768 is GPT2-small hidden size
        
        # State processing
        self.state_encoder = nn.Sequential(
            nn.Linear(3, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        # Action head (mean and log_std for each action)
        self.action_mean = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 4),  # 4 actions: forward, left, right, interact
            nn.Tanh()
        )
        
        self.action_log_std = nn.Parameter(torch.zeros(1, 4))
        
    def forward(self, vision, command_embedding, robot_state):
        # Process each input stream
        vision_features = self.vision_encoder(vision)
        command_features = self.command_encoder(command_embedding)
        state_features = self.state_encoder(robot_state)
        
        # Combine features
        combined = torch.cat([vision_features, command_features, state_features], dim=1)
        
        # Generate action distribution parameters
        mean = self.action_mean(combined)
        std = torch.exp(self.action_log_std)
        
        # Create normal distribution
        dist = Normal(mean, std)
        
        # Sample action
        if self.training:
            action = dist.rsample()  # Use reparameterization trick during training
        else:
            action = mean  # Use mean during evaluation
        
        return dist, action

class RobotBrain(nn.Module):
    """Simplified robot control architecture without LLM for initial training."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        
        # Initialize robot control network
        self.control_network = RobotControlNetwork()
        
        self.to(device)
    
    def forward(self, vision, task_description, robot_state):
        # For now, we'll skip the LLM and just use a fixed command embedding
        batch_size = vision.size(0)
        command_embedding = torch.zeros(batch_size, 768).to(self.device)  # Placeholder command
        
        # Generate robot actions using the control network
        dist, action = self.control_network(vision, command_embedding, robot_state)
        return dist, action, "Simple navigation mode"  # Return distribution, action, and placeholder reasoning
    
    def get_parameters(self):
        """Get all trainable parameters."""
        return self.control_network.parameters() 