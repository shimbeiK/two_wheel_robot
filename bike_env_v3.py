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
    # ハイパーパラメータを取得(from json file)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "parameters_ppo.json")
    with open(json_path, 'r') as f:
        json_param = json.load(f)["dqn_bike"]
    parameters = {"lr": json_param["lr"],
                "gamma": json_param["gamma"],
                "episodes": json_param["episodes"],
                "epsilon": json_param["epsilon"],
                "buffer_size": json_param["buffer_size"],
                "batch_size": json_param["batch_size"],
                "td_interval": json_param["td_interval"],
                "input_size":json_param["input_size"],
                "action_size": json_param["action_size"],
                "max_step": json_param["max_step"]        
    }

    # def __init__(self, model, data, render_mode=None, max_step=parameters["max_step"]):
    def __init__(self, xml_path, render_mode=None, max_step=parameters["max_step"]):
        # super(StandingEnv(), self).__init__(gym.env)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # self.model = model
        # self.data = data
        self.step_count = 0
        self.l_wheel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "tire_back_pitch")
        self.prev_angular_vel = 0.0
        self.prev_action = [0.0, 0.0]
        self.wheel_pos = self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]]
        # self.data.qpos[8] = np.deg2rad(0)
        self.data.qpos[8] = np.deg2rad(-60)
        self.env_cfg, obs_cfg, self.reward_cfg, command_cfg = PythonConfig.get_cfg()


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
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        # self.action_space = spaces.Box(-self.MAX_TORQUE, self.MAX_TORQUE, dtype=np.float32)

        # 観測空間：位置、速度、角度、角速度
        high = np.array([self.ANGLE_THRESHOLD, np.finfo(np.float32).max, 
                         np.finfo(np.float32).max, 1, 1, 1, 1,
                         self.POS_THRESHOLD, self.POS_THRESHOLD], dtype=np.float32)
                        #  np.finfo(np.float32).max, self.MAX_ANGLE, self.MAX_TORQUE], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    # センサから観測する。位置はMujoco環境から得る
    def _get_obs(self, action_steer, action_back, prev_action_steer, prev_action_back):
        rotmat = self.data.xmat[1].reshape(3, 3)
        rot = R.from_matrix(rotmat)
        angle = rot.as_euler('xyz', degrees=False)
        imu = np.rad2deg(angle)  # Convert to radians
        angular_vel = self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]]
        angular_acc = (angular_vel - self.prev_angular_vel) / (self.model.opt.timestep * self.frame_skip)   
        # Update previous value for next loop
        self.prev_angular_vel = angular_vel
        body_pos_x = self.data.qpos.copy()[0]
        body_pos_y = self.data.qpos.copy()[1]

        # Ac_motor_vel = self.data.sensordata[self.adr_F_vel]  # 0〜2πに正規化 
        # print(np.deg2rad(imu[0])) 
        return np.array([np.deg2rad(imu[0]), angular_vel, angular_acc, action_steer, 
                         action_back, prev_action_steer, prev_action_back, 
                         body_pos_x, body_pos_y], dtype=np.float32)

    # バイクの傾きと位置の変化から報酬を決定
    def _reward(self, obs):
        imu, angular_vel, angular_acc, action_steer, action_back, prev_action_steer, prev_action_back, body_pos_x, body_pos_y = obs
        reward = self.reward_cfg["survival_bonus"]  # 生存ボーナス（時間経過に対する報酬）

        reward += self.reward_cfg["upright_posture"] * (np.deg2rad(45) - abs(imu)) / np.deg2rad(45)
        reward += self.reward_cfg["pos_penalty"] * np.sqrt(body_pos_x**2 + body_pos_y**2) / np.sqrt(self.POSX_THRESHOLD**2 + self.POSY_THRESHOLD**2)  # 位置のペナルティ（中心からの距離に比例）
        reward += self.reward_cfg["angular_vel_penalty"] * abs(angular_vel)
        reward += self.reward_cfg["steering_change_penalty"] * max(2, abs(action_steer - prev_action_steer)) / 2  # 急激なステアリング変化を抑制
        reward += self.reward_cfg["torque_change_penalty"] * max(2, abs(action_back - prev_action_back)) / 2    # 急激な後輪トルク変化を抑制
        # reward += 0.5 * (1 - action_steer) # ステアリングの使用を抑制
        # reward -= 1 * abs(action_back) # 後輪トルクの使用を抑制

        return reward

    def step(self, action):
        # print(action)
        action_angle = action[0]*self.MAX_STEER
        action_torque = action[1]*self.MAX_TORQUE
        if self.env_cfg["action_noise"] == True:
            action_angle += np.random.normal(0, 1) * self.MAX_STEER / 100  # ステアリングにノイズを加える
            action_torque += np.random.normal(0, 1) * self.MAX_TORQUE / 100  # トルクにノイズを加える
        # print(action)
        self.data.ctrl[0] = action_angle
        self.data.ctrl[1] = action_torque
        for i in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            # time.sleep(0.002)
                
        obs = self._get_obs(action[0], action[1], self.prev_action[0], self.prev_action[1])
        self.prev_action = [action[0], action[1]]
        reward = self._reward(obs)

        terminated = bool(abs(obs[0]) > self.ANGLE_THRESHOLD 
                          or abs(obs[7]) > self.POSX_THRESHOLD
                          or abs(obs[8]) > self.POSY_THRESHOLD
                          )
        truncated = bool(self.step_count >= self.MAX_STEP)
        if truncated or terminated:
            if self.step_count > 100:
                reward += 10.0  # 倒れたら大きくペナルティ
        # print("obs:", obs[0], "terminated:", terminated, "truncated:", truncated)
        # 辞書の中にデータを入れる
        info = {
            # "mj_data": self.data, 
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
        return self._get_obs(self.env_cfg["initial_steer_deg"], 
                             self.env_cfg["initial_torque"], 0.0, 0.0), {}

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
