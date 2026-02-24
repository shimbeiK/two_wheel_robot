import mujoco
import mujoco.viewer
import time
import numpy as np
from stable_baselines3 import PPO
from bike_env_v3 import StandingEnv

# --- 1. Setup the Environment for Evaluation ---
# We use ONE environment with render_mode="human"
MODEL_PATH = "mjcf2/scene.xml"
m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)

# Important: Set render_mode="human" here to open the window
# env = StandingEnv(m, d, render_mode="human")
env = StandingEnv(xml_path=MODEL_PATH, render_mode="human")
# env = VecNormalize.load(stats_load_path, env)
# # 【重要】推論時は統計情報を更新せず、報酬の正規化も行わない設定にする
# env.training = False
# env.norm_reward = False
l_wheel_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "tire_back_pitch")
# prev_angular_vel = env.data.qvel[env.model.jnt_dofadr[l_wheel_id]]
angular_vel = env.data.sensor("imu_gyro").data.copy()[0]+np.random.normal(0, 0.01)  # ジャイロのx軸の角速度にノイズを加える

# --- 2. Load the Trained Model ---
# Load the zip file you just saved
# model = PPO.load("two_wheel_robot/results/stop_withCon/20260214-1804_continue", env=env)
model = PPO.load("two_wheel_robot/results/stop_withCon/20260223-2335_eval/best_model", env=env)
# model = PPO.load("two_wheel_robot/results/stop_withCon/kourin25/20260222-1110_con.zip", env=env)
# model = PPO.load("two_wheel_robot/results/stop_ignoreCon/ppo_standing_20260214-1623.zip", env=env)
# --- 3. Run the Simulation Loop ---
obs, _ = env.reset()
print("Running trained model... Press Ctrl+C to stop.")
counter = 0
while True:
    counter+=1
    print(counter)
    # Predict the action (deterministic=True gives the 'best' action, False is slightly random)
    action, _ = model.predict(obs, deterministic=True)
    prev_angular_vel = env.data.qvel[env.model.jnt_dofadr[l_wheel_id]]
    # prev_angular_vel = env.data.sensor("imu_gyro").data.copy()[0]+np.random.normal(0, 0.01)  # ジャイロのx軸の角速度にノイズを加える
    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    # print("action:", action)
    # print("pos", obs[7], obs[8])
    # print("action:", np.rad2deg(action[0]), action[1])
    print("imu", np.rad2deg(obs[0]))
    print("angular_vel:", prev_angular_vel)

    # Render is often handled automatically by render_mode="human" in gymnasium,
    # but we call it here just in case your custom env requires it.
    env.render()
    
    # Slow down slightly to match real-time (optional, depends on your PC speed)
    time.sleep(0.002)

    if terminated or truncated:
        obs, _ = env.reset()
        counter = 0