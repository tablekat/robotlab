# 2D Robot Simulation with LLM-based Control

This project implements a 2D robot simulation where the robot's behavior is controlled by a combination of a small pre-trained Language Model (LLM) and a specialized neural network for robot control. The system uses GRPO (Generalized Proximal Policy Optimization) for reinforcement learning to train both networks together.

On the way to training, I'd also like to try to develop chain-of-thought reasoning in the LLM for more complex tasks.

## Features

- 2D physics-based robot simulation using Pygame
- Small-scale LLM integration for high-level decision making
- Custom neural network for low-level robot control
- Chain-of-thought reasoning for complex task planning
- GRPO-based reinforcement learning pipeline
- Simple tasks mirroring real-world chores

## Task Examples

1. **Room Navigation**: Navigate to specific locations while avoiding obstacles
2. **Object Collection**: Gather and sort objects into designated areas
3. **Sequential Tasks**: Complete multi-step tasks like "collect red items, then move them to the blue zone"
4. **Conditional Actions**: Respond to environmental changes (e.g., "if path blocked, find alternative route")

## Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `src/`
  - `environment/`: 2D simulation environment
  - `models/`: Neural network architectures
  - `training/`: Training pipeline and GRPO implementation
  - `utils/`: Helper functions and utilities

## Usage

1. To run the simulation:
```bash
python src/main.py
```

2. To start training:
```bash
python src/train.py
```

## Architecture

The system combines two main components:
1. A small-scale LLM (based on GPT-2 small) for high-level planning
2. A robot control network for converting plans into actions

The LLM provides chain-of-thought reasoning for task planning, while the control network translates these plans into concrete robot movements in the 2D space.

## Training

The training process uses GRPO to optimize both networks simultaneously:
- The LLM learns to generate better task plans
- The control network learns to execute these plans effectively
- Both networks are rewarded for successful task completion

## License

MIT License 