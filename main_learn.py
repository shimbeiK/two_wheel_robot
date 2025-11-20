from stable_baselines3 import PPO
from bike_env import StandingEnv
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: StandingEnv()])

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs={
        "net_arch": [64, 64],
        "log_std_init": 0.3
    },
    learning_rate=3e-4,
    n_steps=2048,                # ← 増やすと学習安定
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,              # ← 緩めに探索させる
    normalize_advantage=True,    # ← Trueにすべし
    verbose=1
)

model.learn(total_timesteps=300000)
model.save("ppo_standing")