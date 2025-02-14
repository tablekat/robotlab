import torch
import time
from environment.env import RobotEnv
from models.robot_brain import RobotBrain

def main():
    # Create environment
    env = RobotEnv()
    
    # Initialize robot brain
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    robot_brain = RobotBrain(device=device)
    
    # Load trained model
    checkpoint = torch.load('checkpoint_1000.pt', map_location=device)
    robot_brain.load_state_dict(checkpoint['robot_brain_state_dict'])
    robot_brain.eval()
    
    # Example tasks to try
    tasks = [
        "Navigate to the nearest red target while avoiding obstacles",
        "Collect all blue objects and bring them to the green zone",
        "Patrol the perimeter of the room",
        "Find the largest open area and stay there",
        "Follow the path marked by yellow dots"
    ]
    
    print("Robot brain loaded! Starting simulation...")
    print("\nAvailable tasks:")
    for i, task in enumerate(tasks, 1):
        print(f"{i}. {task}")
    
    while True:
        try:
            task_idx = int(input("\nEnter task number (or 0 to quit): ")) - 1
            if task_idx == -1:
                break
            if 0 <= task_idx < len(tasks):
                run_task(env, robot_brain, tasks[task_idx])
            else:
                print("Invalid task number!")
        except ValueError:
            print("Please enter a valid number!")

def run_task(env, robot_brain, task):
    print(f"\nExecuting task: {task}")
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Convert state to tensor
        vision = torch.FloatTensor(state['vision']).unsqueeze(0).to(robot_brain.device)
        robot_state = torch.FloatTensor(state['robot_state']).unsqueeze(0).to(robot_brain.device)
        
        # Get action from policy
        with torch.no_grad():
            action, reasoning = robot_brain(vision, task, robot_state)
            action = action.cpu().numpy()[0]
        
        # Print robot's reasoning
        print("\nRobot's thought process:")
        print(reasoning)
        
        # Take action in environment
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state
        
        # Add small delay for visualization
        time.sleep(0.1)
    
    print(f"\nTask completed! Total reward: {total_reward:.2f}")

if __name__ == "__main__":
    main() 