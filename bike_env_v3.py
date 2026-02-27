'''
consider with constraint of observation and action space. and input/output noise.
'''

import gymnasium as gym
from gymnasium import spaces
import mujoco, json, time
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from madgwick import MadgwickFilter
from cfg_standing import PythonConfig

class StandingEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    # def __init__(self, model, data, render_mode=None, max_step=parameters["max_step"]):
    def __init__(self, xml_path, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # self.model = model
        # self.data = data
        self.step_count = 0
        self.l_wheel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "tire_back_pitch")
        self.drive_vel = 0.0
        self.prev_action = [0.0, 0.0]
        self.wheel_pos = self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]]
        # self.data.qpos[8] = np.deg2rad(0)
        self.data.qpos[8] = np.deg2rad(-60)
        self.total_odometry = 0.0
        self.prev_odometry = 0.0

        self.env_cfg, obs_cfg, self.reward_cfg, command_cfg = PythonConfig.get_cfgs()


        self.ANGLE_THRESHOLD = self.env_cfg["termination_if_roll_greater_than"]  # radians
        self.POSX_THRESHOLD = self.env_cfg["termination_if_posX_greater_than"]  # meters
        self.POSY_THRESHOLD = self.env_cfg["termination_if_posY_greater_than"]  # meters
        self.MAX_STEP = self.env_cfg["termination_if_step_count_greater_than"]
        self.frame_skip = self.env_cfg["frame_skip"]

        # 1. render_mode を保存する
        self.render_mode = render_mode

        # 2. レンダラーとビューワーは初期値 None (使う時に作成する "Lazy initialization")
        self.viewer = None
        self.renderer = None        
        self.MAX_TORQUE = self.env_cfg["drive_torque_scale"]  # 最大トルク
        self.MAX_STEER = self.env_cfg["steering_angle_scale"]
        self.NOISE_ANGLE = self.env_cfg["noise_angle"]  # 初期傾きのノイズ幅（±deg）
        self.action_space = spaces.Box(-1.0, 1.0, dtype=np.float32)
        # self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        # self.action_space = spaces.Box(-self.MAX_TORQUE, self.MAX_TORQUE, dtype=np.float32)

        # 観測空間：位置、速度、角度、角速度
        high = np.array([self.ANGLE_THRESHOLD, np.finfo(np.float32).max, 
                         np.finfo(np.float32).max, np.finfo(np.float32).max]) # 角度、角速度、角加速度、前回のアクション、前々回のアクション、オドメトリ
                        #  self.POSX_THRESHOLD, self.POSY_THRESHOLD], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    # センサから観測する。位置はMujoco環境から得る
    def _get_obs(self):
        rotmat = self.data.xmat[1].reshape(3, 3)
        rot = R.from_matrix(rotmat)
        angle = rot.as_euler('xyz', degrees=False)
        imu = np.rad2deg(angle)  # Convert to radians
        angular_vel = angular_vel = self.data.sensor("imu_gyro").data.copy()[0]
        drive_vel = self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]]
        # Update previous value for next loop
        self.drive_vel = drive_vel

        # --- FIX: Ensure the actions are flat numbers (scalars), not arrays ---
        self.total_odometry += drive_vel * 3.1 * self.model.opt.timestep * self.frame_skip

        return np.array([np.deg2rad(imu[0]), angular_vel, drive_vel,
                         self.total_odometry], dtype=np.float32)
                        #  act, prev_act, body_pos_x, body_pos_y], dtype=np.float32)

    # バイクの傾きと位置の変化から報酬を決定
    def _reward(self, obs):
        # imu, angular_vel, angular_acc, action_back, prev_action_back, body_pos_x, body_pos_y = obs
        imu, angular_vel, action_back, total_odometry = obs
        reward = self.reward_cfg["survival_bonus"]  # 生存ボーナス（時間経過に対する報酬）
        reward += self.reward_cfg["upright_posture"] * (np.deg2rad(45) - abs(imu)) / np.deg2rad(45)
        reward += (abs(self.prev_odometry) - abs(total_odometry)) * self.reward_cfg["odometry_penalty"]
        self.prev_odometry = total_odometry

        return reward


    def step(self, action):
        # print(action)
        action_angle = action[0]*self.MAX_STEER
        action_torque = action[1]*self.MAX_TORQUE
        if self.env_cfg["action_noise"] == True:
            action_angle += np.random.normal(0, 1) * self.MAX_STEER * self.env_cfg["action_noise_range"]  # ステアリングにノイズを加える
            action_torque += np.random.normal(0, 1) * self.MAX_TORQUE * self.env_cfg["action_noise_range"]  # トルクにノイズを加える
        # print(action)
        self.data.ctrl[0] = action_angle
        self.data.ctrl[1] = action_torque
        for i in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            # time.sleep(0.002)
                
        obs = self._get_obs()
        reward = self._reward(obs)

        terminated = bool(abs(obs[0]) > self.ANGLE_THRESHOLD)
        truncated = bool(self.step_count >= self.MAX_STEP)
        # if truncated:
        #     reward += self.reward_cfg["reward_if_truncated"]  # 倒れたら大きくペナルティ
        # 辞書の中にデータを入れる
        info = {
            # "imu": obs[0], 
            # "other_info": 123
        }
        self.step_count += 1

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[8] = self.env_cfg["initial_steer_deg"]
        angle = self.env_cfg["initial_tilt_deg"]
        if self.env_cfg["init_noise"] == True:
            angle += np.random.normal(0, np.deg2rad(self.NOISE_ANGLE))
        self.data.ctrl[0] = self.env_cfg["initial_steer_deg"]
        self.data.ctrl[1] = self.env_cfg["initial_torque"]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mju_euler2Quat(self.data.qpos[3:7], np.array([np.deg2rad(angle), 0, 0]), "xyz")    

        self.step_count = 0
        self.wheel_pos = self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]]
        self.prev_angular_vel = 0.0
        self.prev_action = [0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

        # B. rgb_array モード: 画像データ (numpy array) を返す
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                # オフスクリーンレンダラーを作成
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            
            # レンダラーに現在の物理状態を反映してピクセルを読み込む
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        
    def close(self):
        # ウィンドウやリソースの解放
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

if __name__ == "__main__":
    print("実行するファイルが違うで。")