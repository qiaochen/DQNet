from unityagents import UnityEnvironment
import numpy as np
from tqdm import tqdm
from agent import DQAgent
from utils import draw

unity_environment_path = "./Banana_Linux/Banana.x86_64"
best_model_path = "./best_model.checkpoint"

if __name__ == "__main__":
    # prepare environment
    env = UnityEnvironment(file_name=unity_environment_path)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    # number of actions
    action_size = brain.vector_action_space_size

    # examine the state space 
    state = env_info.vector_observations[0]
    state_size = len(state)
    
    agent = DQAgent(state_size, action_size)
    agent.load(best_model_path)

    test_scores = []
    for i_episode in tqdm(range(1, 101)):
        score = 0                                          # initialize the score
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        while True:
            action = agent.act(state)                     # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step

            if done:                                       # exit loop if episode finished
                test_scores.append(score)
                break
                
    avg_score = sum(test_scores)/len(test_scores)
    print("Test Score: {}".format(avg_score))
    draw(test_scores, "./test_score_plot.png", "Test Scores of 100 Episodes (Avg. score {})".format(avg_score))
    env.close()
