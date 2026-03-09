from stable_baselines3 import SAC  # PPOからSACに変更
from bike_env_v3 import StandingEnv
from stable_baselines3.common.monitor import Monitor
import mujoco, time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

MODEL_PATH = "mjcf2/scene.xml"
log_dir = "tboard_logs/sac_clip01"
timestamp = time.strftime("%Y%m%d-%H%M")

# SACは通常、単一環境（n_envs=1）でも十分効率的ですが、並列化も可能です。
# ただし、VecEnvを使う場合はサンプル効率に注意してください。
env = make_vec_env(
    StandingEnv, 
    n_envs=4, # SACは通常1〜4環境程度で十分なことが多いです
    env_kwargs={"xml_path": MODEL_PATH, "render_mode": None}
)

eval_env = make_vec_env(
    StandingEnv, 
    n_envs=1, 
    env_kwargs={"xml_path": MODEL_PATH, "render_mode": None}
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"two_wheel_robot/results/sac_results/{timestamp}_eval",
    eval_freq=10000,
    deterministic=True,
    render=False
)

# SACモデルの設定
sac_model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    device="cuda", # GPUが使えるなら"cuda"推奨
    learning_rate=3e-4,
    buffer_size=1000000,  # リプレイバッファのサイズ
    learning_starts=1000, # 学習を開始するまでのステップ数（最初はランダム行動）
    batch_size=256,       # 1回の更新で使用するサンプル数
    tau=0.005,            # ターゲットネットワークのソフト更新係数
    gamma=0.99,
    train_freq=5,         # 1ステップごとに学習を行う
    gradient_steps=2,     # 1ステップにつき何回勾配更新を行うか
    ent_coef="auto",      # エントロピー係数の自動調整（SACの肝）
    target_update_interval=1,
    policy_kwargs={
        "net_arch": [256, 256], # SACは少し深めのネットワークが好まれます
        "log_std_init": -1.2,     # 初期探索範囲の調整
    },
    tensorboard_log=log_dir
)

# 学習開始
sac_model.learn(
    total_timesteps=3000000, 
    callback=eval_callback,
    progress_bar=True # 学習状況が見やすくなります
)

sac_model.save(f"two_wheel_robot/results/sac_results/bike_sac_{timestamp}")