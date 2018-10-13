# Project Report

The latest version of this project has used the following extensions to improve the original Deep Q-Network:
- Double DQNs
- Dueling DQN (aka DDQN)


## Learning Algorithm

- Network Architecture
  ![img](https://raw.githubusercontent.com/qiaochen/DQNet/master/network_architecture.jpg)
The input state vector is encoded by 2 fully connected layers befere branching into the `state value` and the `advantage` streams, which is according to the Dueling DQN scheme.
The scalar generated through the state value stream (representing `State Value`) then joins in the computation with the output of the `Advantage` stream according to the equation, and yields the final action values.


- Hyper-parameters
  - batch size = 128
  - replay capacity = 10000
  - gamma = 99 
  - alpha = 0.2 
  - epsilon = 0.2
  - epsilon decay rate = 0.99999
  - minimal epsilon = .02
  - target q network update interval = 10
  - learning rate = 5e-5
  - learning rate decay rate = 0.95
- Training Strategy
  - Adam is used as the optimizer, a `L2 regularization` strength of `1e-3` is set to counteract overfitting
  - An `early-stop` scheme is applied to stop training if the 100-episode-average score continues decreasing over `10` consecutive episodes.
  - Each time the model gets worse regarding avg scores, the model recovers from the last best model and the learning rate of Adam is decreased: `new learning rate = old learning rate * learning rate decay rate` 
  - Gradients are clampped into `(-1, +1)` range to prevent exploding
  - Used `batch normalization` to facilitate optimization

## Performance Evaluation
### Training
During training, the best performance was achieved in less than 350 episodes. The episodic and average (over 100 latest episodes) scores are plotted as following:
- Reward per-episode during training
![img](https://raw.githubusercontent.com/qiaochen/DQNet/master/training_score_plot.png)
- Average reward over latest 100 episodes during training
![img](https://raw.githubusercontent.com/qiaochen/DQNet/master/training_100avgscore_plot.png)
As can be seen from the plot, the average score gradually reached around 16.0 during training, before the early-stopping scheme terminates the training process.

### Testing
The scores of 100 testing episodes are visualized as follows:
![img](https://raw.githubusercontent.com/qiaochen/DQNet/master/test_score_plot.png)
The model obtained an average score of 16.31 during testing.

## Conclusion
The trained model has successfully sovled the navigation task. The performance:
1. an average score of `16.31` over `100` episodes 
2. the best model was trained in less than `350` episodes

has fulfilled the passing threshold of solving the problem: obtain an average score of `13.00` over `100` testing episodes.

## Ideas for Future Work

- Using Prioritized Experience Replay, so that valuable experience can be re-used better.
- Using raw pixels input
- TD(X): currently TD-1 is applied, how about using more sampled steps? (the other extreme is MC method) .
