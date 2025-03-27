from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_env import SnakeGameEnv


class LRSchedule:
    def __init__(self, total_timesteps, decay_start=0.5):
        self.total_timesteps = total_timesteps
        self.decay_start = 1 - decay_start
        self.timesteps_trained = 0

    def __call__(self, _):
        self.timesteps_trained += 1
        progress_remaining = 1 - self.timesteps_trained / self.total_timesteps

        if progress_remaining < self.decay_start:
            return 3e-4 * (progress_remaining / self.decay_start)
        return 3e-4


log_dir = "logs"

vec_env = make_vec_env(lambda: SnakeGameEnv(num_snakes=1, num_teams=1), n_envs=32)
# env = SnakeGameEnv(num_snakes=1, num_teams=1)



# model = PPO('MultiInputPolicy', vec_env, verbose=True, device='cuda', tensorboard_log=log_dir, n_steps=128, batch_size=2048, learning_rate=0.0003)
model = PPO('CnnPolicy', vec_env, verbose=True, device='cuda', tensorboard_log=log_dir, n_steps=128, batch_size=2048, learning_rate=0.0003)
# # model = PPO.load("ppo_snake", env=env, device="cuda", tensorboard_log=log_dir)


for i in range(100):
    model.learn(100000, progress_bar=True, tb_log_name="PPO-3.31", reset_num_timesteps=False)
    model.save('ppo_snake3.31.zip')

