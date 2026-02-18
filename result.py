import mujoco
import mujoco.viewer
import time
import numpy as np
from stable_baselines3 import PPO
from bike_env_v2 import StandingEnv

# --- 1. Setup the Environment for Evaluation ---
# We use ONE environment with render_mode="human"
MODEL_PATH = "mjcf2/scene.xml"
m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)

# Important: Set render_mode="human" here to open the window
# env = StandingEnv(m, d, render_mode="human")
env = StandingEnv(xml_path=MODEL_PATH, render_mode="human")
# l_wheel_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "tire_top_pitch")
# prev_angular_vel = env.data.qvel[env.model.jnt_dofadr[l_wheel_id]]

# --- 2. Load the Trained Model ---
# Load the zip file you just saved
# model = PPO.load("two_wheel_robot/results/stop_withCon/20260214-1804_continue", env=env)
model = PPO.load("two_wheel_robot/results/stop_withCon/20260218-1741_eval/best_model", env=env)
# model = PPO.load("two_wheel_robot/results/stop_withCon/kourin2520260217-2230_con.zip", env=env)
# model = PPO.load("two_wheel_robot/results/stop_ignoreCon/ppo_standing_20260214-1623.zip", env=env)
# --- 3. Run the Simulation Loop ---
obs, _ = env.reset()
print("Running trained model... Press Ctrl+C to stop.")

while True:
    # Predict the action (deterministic=True gives the 'best' action, False is slightly random)
    action, _ = model.predict(obs, deterministic=True)
    # prev_angular_vel = env.data.qvel[env.model.jnt_dofadr[l_wheel_id]]

    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    print("action:", action)
    # print("angular_vel:", prev_angular_vel)

    # Render is often handled automatically by render_mode="human" in gymnasium,
    # but we call it here just in case your custom env requires it.
    env.render()
    
    # Slow down slightly to match real-time (optional, depends on your PC speed)
    time.sleep(0.002)

    if terminated or truncated:
        obs, _ = env.reset()