import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
 
class StandingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
 
    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path('two_wheel_robot/scene.xml')
        self.data = mujoco.MjData(self.model)
 
        self.render_mode = render_mode
        self.viewer = None
 
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -20.0, -50.0], dtype=np.float32),
            high=np.array([np.pi, 20.0, 50.0], dtype=np.float32),
            dtype=np.float32
        )
 
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )
 
        self.joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_hinge")
        self.qvel_index = self.model.jnt_dofadr[self.joint_id]
        self.np_random = np.random.default_rng()
 
    def quat_to_pitch(self, q):
        w, x, y, z = q
        sinp = 2.0 * (w * y - z * x)
        cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(sinp, cosp)
 
    def _get_obs(self):
        quat = self.data.qpos[3:7]
        theta_y = self.quat_to_pitch(quat)
        omega_y = self.data.qvel[4]
        wheel_speed = self.data.qvel[self.qvel_index]
        return np.array([theta_y, omega_y, wheel_speed], dtype=np.float32)
 
    def step(self, action):
        # PPOの出力（[-1, 1]）をスケールしてMuJoCoに渡す
        torque_scale = 1  # ← ここだけ調整すればOK（±3.0トルク想定）
        scaled_action = np.clip(action[0], -1.0, 1.0) * torque_scale
        self.data.ctrl[0] = float(scaled_action)
 
        mujoco.mj_step(self.model, self.data)
 
        obs = self._get_obs()
        theta, omega, wheel_speed = obs
 
        self.timestep_count += 1 
        reward = (
            - 5.0 * theta**2
            - 0.1 * omega**2
            #- 0.001 * wheel_speed**2
            #- 0.1 * (action[0]**2)       # ← コレがトルク抑制項
            + 5.0 * (abs(theta) < 0.005)
            - 0.5 * (np.sign(theta) != np.sign(wheel_speed))  # 
            #+ 0.1 * self.timestep_count   # ← ここで時間に比例して報酬を加算
        )
 
        terminated = bool(abs(theta) > np.pi / 6)  # 30度以上で終了
        truncated = False
 
        return obs, reward, terminated, truncated, {}
 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
 
        # 位置も完全にリセット
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
 
        self.timestep_count = 0
 
 
       # ランダムなY軸傾き（±0.3rad）
        theta = self.np_random.uniform(low=-0.05, high=0.05)
        half_theta = theta / 2
        self.data.qpos[3] = np.cos(half_theta)  # w
        self.data.qpos[5] = np.sin(half_theta)  # y
 
        # Y軸の角速度（±1.0rad/s）
        omega_y = self.np_random.uniform(low=-1.0, high=1.0)
        self.data.qvel[3] = omega_y  # qvel[3] = 回転自由度Y軸（free jointの場合）
 
        mujoco.mj_forward(self.model, self.data)  # 物理整合性の更新
 
        return self._get_obs(), {}
 
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
 
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None