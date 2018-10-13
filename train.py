from unityagents import UnityEnvironment
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

    num_episodes = 10000
    agent = DQAgent(state_size, 
                    action_size,
                    batch_size=128,
                    replay_capacity=10000, 
                    gamma=.99, 
                    alpha=.2, 
                    eps=.2,
                    eps_decay=.99999,
                    eps_min = .02,
                    target_update_interval=10,
                    learning_rate=5e-5,
                    lr_decay=.95)
    total_rewards = []
    avg_scores = []
    max_avg_score = -1
    worsen_tolerance = 10  # for early-stopping training if consistently worsen for # episodes
    for i_episode in range(1, num_episodes+1):
        env_inst = env.reset(train_mode=True)[brain_name]   # reset the environment
        state = env_inst.vector_observations[0]             # get the current state
        score = 0                                           # initialize the score
        done = False
        while not done:
            action = agent.sample_action(state)             # select an action
            env_inst = env.step(action.item())[brain_name]  # send the action to the environment
            next_state = env_inst.vector_observations[0]    # get the next state
            reward = env_inst.rewards[0]                    # get the reward
            done = env_inst.local_done[0]                   # see if episode has finished
            agent.update_model(state, action, next_state, reward)
            score += reward                                 # update the score
            state = next_state                              # roll over the state to next time step
            
        total_rewards.append(score)
        print("Episodic Score: {}".format(score))
        
        if len(total_rewards) >= 100:                       # record avg score for the latest 100 steps
            latest_avg_score = sum(total_rewards[(len(total_rewards)-100):]) / 100
            print("100 Episodic Everage Score: {}".format(latest_avg_score))
            avg_scores.append(latest_avg_score)
          
            if max_avg_score <= latest_avg_score:           # record better results
                worsen_tolerance = 10                       # re-count tolerance
                max_avg_score = latest_avg_score
                if max_avg_score > 10:                      # saving tmp model
                    agent.save(best_model_path)
            else:                                           
                worsen_tolerance -= 1                       # count worsening counts
                if max_avg_score > 10:                      # continue from last best-model
                    agent.load(best_model_path)
                if worsen_tolerance <= 0:                   # earliy stop training
                    print("Early Stop Training.")
                    break
                    
    draw(total_rewards,"./training_score_plot.png", "Training Scores (Per Episode)")
    draw(avg_scores,"./training_100avgscore_plot.png", "Training Scores (Average of Latest 100 Episodes)", ylabel="Avg. Score")
    env.close()

                    
