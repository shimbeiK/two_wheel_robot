from stable_baselines3 import PPO
from bike_env_v2 import StandingEnv
# from segway_env import StandingEnv
from stable_baselines3.common.monitor import Monitor
import mujoco, time
import mujoco.viewer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

class RenderCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        # 毎ステップ呼び出される
        self.env.render()
        return True
    
MODEL_PATH = "mjcf2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
log_dir = "tboard_logs/clip02"
timestamp = time.strftime("%Y%m%d-%H%M")
# 接触全体を無効化
# model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
# for i, p in enumerate(model.pair_geom1):
#     g1 = model.pair_geom1[i]
#     g2 = model.pair_geom2[i]

#     # 例: geom1="wheel", geom2="body" のペアだけ無効化
#     if model.geom(g1).name == "tire_holder" and model.geom(g2).name == "tire_top":
#         model.pair_contype[i] = 0    # 0 → 接触生成しない

# env = DummyVecEnv([lambda: Monitor(StandingEnv(model, data, render_mode="human"))])
# env = make_vec_env(lambda: Monitor(StandingEnv(model, data, render_mode="rgb_array")), n_envs=1)    
# env = make_vec_env(lambda: Monitor(StandingEnv(model, data)), n_envs=16)    
env = make_vec_env(
    StandingEnv, 
    n_envs=32, 
    env_kwargs={"xml_path": MODEL_PATH, 
    "render_mode": None})

eval_env = make_vec_env(
    StandingEnv, 
    n_envs=1, 
    env_kwargs={"xml_path": MODEL_PATH, 
    "render_mode": None})

render_callback = RenderCallback(env)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"two_wheel_robot/results/stop_withCon/{timestamp}_eval",
    eval_freq=5000,
    deterministic=True,
    render=False
)

ppo_model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs={
        "net_arch": [64, 64],
        "log_std_init": 0.3
    },
    device="cpu",  
    learning_rate=3e-4,
    n_steps=8192,                # ← 増やすと学習安定
    batch_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.4,              # ← better
    # clip_range=0.5,            # ← 緩めに探索させる
    normalize_advantage=True,    # ← Trueにすべし
    verbose=1,
    tensorboard_log=log_dir
)

# ppo_model = PPO.load("two_wheel_robot/results/stop_withCon/kourin2520260217-2230_con.zip",
#                      device="cpu",clip_range=0.2, env=env)
# ppo_model = PPO.load("two_wheel_robot/results/terning", env=env, clip_range=0.2)
# XMLファイルを読み込み
# MODEL_PATH = "sim_env/bike.xml"
time.sleep(0.1)
# ppo_model.learn(total_timesteps=5000000, callback=eval_callback, reset_num_timesteps=False)
ppo_model.learn(total_timesteps=10000000, callback=eval_callback)

ppo_model.save(f"two_wheel_robot/results/stop_withCon/kourin25{timestamp}_con")