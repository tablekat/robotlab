import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

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
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 4),  # 4 actions: forward, left, right, interact
            nn.Tanh()
        )
        
    def forward(self, vision, command_embedding, robot_state):
        # Process each input stream
        vision_features = self.vision_encoder(vision)
        command_features = self.command_encoder(command_embedding)
        state_features = self.state_encoder(robot_state)
        
        # Combine features
        combined = torch.cat([vision_features, command_features, state_features], dim=1)
        
        # Generate actions
        actions = self.action_head(combined)
        return actions

class RobotBrain(nn.Module):
    """Combined LLM and robot control architecture."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        
        # Initialize small GPT2 model with LoRA fine-tuning
        self.llm_config = GPT2Config(
            n_layer=6,
            n_head=8,
            n_embd=768,
            vocab_size=50257,
            use_cache=True
        )
        self.llm = GPT2LMHeadModel(self.llm_config)
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add LoRA adapters for efficient fine-tuning
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        self.llm = get_peft_model(self.llm, peft_config)
        
        # Initialize robot control network
        self.control_network = RobotControlNetwork()
        
        self.to(device)
    
    def forward(self, vision, task_description, robot_state):
        # Generate plan using LLM with chain-of-thought prompting
        # Handle batched inputs by taking the first item in the batch
        x = robot_state[0, 0].item()
        y = robot_state[0, 1].item()
        orientation = robot_state[0, 2].item()
        
        prompt = f"""Task: {task_description}
Current state: The robot is at position ({x:.1f}, {y:.1f}) with orientation {orientation:.1f}.
Let's think about this step by step:
1) First, I should..."""
        
        # Generate chain of thought reasoning
        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate text
            generated_ids = self.llm.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            reasoning = self.tokenizer.decode(generated_ids[0])
            
            # Get command embedding from the last hidden state
            model_outputs = self.llm(**inputs, output_hidden_states=True)
            command_embedding = model_outputs.hidden_states[-1][:, -1]
        
        # Generate robot actions using the control network
        actions = self.control_network(vision, command_embedding, robot_state)
        return actions, reasoning
    
    def get_llm_parameters(self):
        """Get LLM parameters for training."""
        return self.llm.parameters()
    
    def get_control_parameters(self):
        """Get control network parameters for training."""
        return self.control_network.parameters() 