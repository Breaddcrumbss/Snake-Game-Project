from stable_baselines3.common.evaluation import evaluate_policy
from gym_env import SnakeGameEnv
from stable_baselines3 import PPO
import torch
import json
import numpy as np
from stable_baselines3.common.monitor import Monitor
import imageio

# action_map = {
#     0: 'stay',
#     1: 'left',
#     2: 'right'
# }

action_map = {
            0: 'up',
            1: 'down',
            2: 'left',
            3: 'right'
        }

with open("param_configs/eval.json", "r") as f:
    game_params = json.load(f)

# Load the trained model 
model = PPO.load("final_model.zip")
# model = PPO.load("models\\old_models2\\ppo_snake4.2_2b.zip")
print(model.policy)
# input("Press Enter to continue...")


# Create a new environment instance for evaluation
env = (SnakeGameEnv(**game_params, render_mode='human'))

# Evaluate the model
# rew, std = evaluate_policy(model, env, n_eval_episodes=50, render=False, return_episode_rewards=False, warn=True, deterministic=False)

frames = []
# For getting the explicit actions probabilities, could be good for data and reporting
num_episodes = 1
all_action_probs = []

scores = []
for _ in range(num_episodes):
    # break
    obs, _ = env.reset()
    done = False

    while not done:
        with torch.no_grad():
            tensor_obs, _ = model.policy.obs_to_tensor(obs)
            action_dist = model.policy.get_distribution(tensor_obs)
            action_probs = torch.exp(action_dist.distribution.logits)[0].tolist()
            all_action_probs.append(action_probs)

        paired_probs = [(action_map[i], round(prob, ndigits=3)) for i, prob in enumerate(action_probs)]
        # print(f"Action Dist: {paired_probs}")
        frame = env.render()
        # frames.append(frame)
        action, _ = model.predict(obs, deterministic=False)
        # print(f"Action: {action_map[int(action)]}")
        obs, reward, done,_, info = env.step(int(action))
        if done:
            print(info)
            scores.append(info['Scores'])
            input("Press Enter to continue...")

print(np.mean(scores))
    # imageio.mimsave("test.gif", frames, fps=30)



# print(f"Mean Reward: {rew:.2f}, Std Reward: {std:.2f}")