# Deep Q-Networks (DQN) for Navigation Control

## Project Details
---
For this project, a DQN agent is trained to navigate (and collect bananas!) in a large, square world.





![img](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif)





### Reward

A `reward` of **+1** is provided for collecting a yellow banana

A `reward` of **-1** is provided for collecting a blue banana. 

Thus, the `goal` of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

### State Space

The `state space` has **37** dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. 

### Action Space

The agent has to learn how to best select the following **4** discrete `actions` to fulfil the goal:

- **0** - move forward.
- **1** - move backward.
- **2** - turn left.
- **3** - turn right.

### Solving the Navigation Problem

The task is episodic, and in order to solve the environment, the agent must get an average score of **+13** over **100** consecutive episodes.

## Getting Started

### Step 1: Clone the Project and Install Dependencies
*Please prepare a python3 virtual environment if necessary.
```
git clone https://github.com/qiaochen/DQNet.git
cd install_requirements
pip install .
```
### Step 2: Download the Unity Environment

For this project, I use the environment provide by **Udacity**. The links to modules at different system environments are copied here for convenience:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

I conducted my experiments in Ubuntu 16.04, so I picked the 1st option.
Then, extract and place the `Banana_Linux` folder within the `project root`.
The project folder structure now looks like this (Program generated .png and model files are excluded):
```
Project Root
     |-install_requirements (Folder)
     |-README.md
     |-agent.py
     |-models.py
     |-train.py
     |-test.py
     |-utils.py
     |-Banana_Linux (Folder)
            |-Banana.x86_64
            |-Banana.x86
            |-Banana_Data (Folder)
```
## Instructions to the Program
### Step 1: Training
```
python train.py
```
After training, the following files will be generated and placed in the project root folder:
- best_model.checkpoint (the trained model)
- training_100avgscore_plot.png (a plot of avg. scores during training)
- training_score_plot.png (a plot of per-episode scores during training)
- unity-environment.log (log file created by Unity)
### Step 2: Test

```
python test.py
```
The testing performance will be summarized in the generated plot within project root:
- test_score_plot.png

