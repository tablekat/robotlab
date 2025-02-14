import torch
from environment.env import RobotEnv
from models.robot_brain import RobotBrain
from training.grpo import GRPO

def main():
    # Create environment
    env = RobotEnv()
    
    # Initialize robot brain
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    robot_brain = RobotBrain(device=device)
    
    # Initialize trainer
    trainer = GRPO(
        robot_brain=robot_brain,
        env=env,
        lr_llm=1e-5,
        lr_control=3e-4,
        batch_size=64,
        n_epochs=10,
        device=device,
        use_wandb=False
    )
    
    # Start training
    print("Starting training...")
    trainer.train(n_iterations=1000)
    print("Training completed!")

if __name__ == "__main__":
    main() 