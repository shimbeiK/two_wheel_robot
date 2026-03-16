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
from cfg_standing_turn import PythonConfig
from collections import deque

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
        self.l_wheel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "back_tire_pitch")
        self.f_wheel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "front_tire_pitch")
        self.wheel_pos = self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]]
        self.alpha = 0.98
        self.env_cfg, obs_cfg, self.reward_cfg, self.cmd_cfg = PythonConfig.get_cfgs()

        self.ANGLE_THRESHOLD = self.env_cfg["termination_if_roll_greater_than"] # radians
        self.POSY_THRESHOLD = self.env_cfg["termination_if_posY_greater_than"]  # meters
        self.MAX_STEP = self.env_cfg["termination_if_step_count_greater_than"]
        self.frame_skip = self.env_cfg["frame_skip"]
        self.DT = self.env_cfg["frame_skip"] * self.model.opt.timestep

        self.latency_step = 3
        self.control_queue = deque([np.zeros(1)] * (self.latency_step + 1), maxlen=self.latency_step+1)
        self.encoder_queue = deque([np.zeros(1)] * (self.latency_step + 1), maxlen=self.latency_step+1)

        # 1. render_mode を保存する
        self.render_mode = render_mode

        # 2. レンダラーとビューワーは初期値 None (使う時に作成する "Lazy initialization")
        self.viewer = None
        self.renderer = None        
        self.MAX_TORQUE = self.env_cfg["drive_torque_scale"]  # 最大トルク
        self.MAX_STEER = self.env_cfg["steering_angle_scale"]
        self.NOISE_ANGLE = self.env_cfg["noise_angle"]  # 初期傾きのノイズ幅（±deg）
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
                
        # 観測空間：位置、速度、角度、角速度
        high = np.array([self.ANGLE_THRESHOLD, np.finfo(np.float32).max, 
                         np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, self.MAX_STEER, 
                         np.finfo(np.float32).max, self.MAX_STEER, np.finfo(np.float32).max], 
                         dtype=np.float32) # 角度、角速度、角加速度、前回のアクション、前々回のアクション、オドメトリ
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    # センサから観測する。位置はMujoco環境から得る
    def _get_obs(self):
        rotmat = self.data.xmat[1].reshape(3, 3)
        rot = R.from_matrix(rotmat)
        angle = rot.as_euler('xyz', degrees=False)
        
        angular_vel = self.data.sensor("body_gyro").data.copy()[0]
        drive_vel = self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]]
        steer_pos = self.data.joint("tire_holder_yaw").qpos[0]
        self.obs = steer_pos
        # --- FIX: Ensure the actions are flat numbers (scalars), not arrays ---
        if(self.env_cfg["real_syncro_noise"] == True):
            angular_vel += np.random.normal(0, np.deg2rad(5.70e-1))
            drive_vel += np.random.normal(0, 1e-1)
            accel = self.data.sensor("body_accel").data
            gyro = self.data.sensor("body_gyro").data           
            accel_roll = np.arctan2(accel[1], accel[2]) + np.random.normal(0, 0.00015)
            gyro_roll = gyro[0] + np.random.normal(0, 0.0008)
            self.filtered_roll = self.alpha * (self.filtered_roll + gyro_roll * self.DT) + (1 - self.alpha) * accel_roll
        else:
            self.filtered_roll = angle[0]

        self.total_odometry += drive_vel * 3.1 * self.DT
        target_drive_vel = self.TARGET_VEL / 0.031
        if(self.TARGET_VEL != 0):
            normalized_diff_vel = (self.TARGET_VEL - 0.031 * drive_vel) / self.TARGET_VEL  # 速度偏差の正規化: 目標速度とのズレを -1.0 ~ 1.0 の範囲に収める
        elif(self.TARGET_VEL == 0):
            normalized_diff_vel = abs(0.031 * drive_vel / self.cmd_cfg["max_vel"])

        delta_theta = self.DT * 0.031 * drive_vel * np.tan(steer_pos) / 0.1605
        self.total_theta += delta_theta
        delta_X = 0.031 * drive_vel * np.cos(steer_pos) * self.DT * np.cos(self.total_theta + delta_theta/2)          
        delta_Y_turn = np.sign(steer_pos) * np.sqrt((0.031 * drive_vel * self.DT)**2 - delta_X**2) * np.cos(self.total_theta)
        # delta_Y = 0.031 * drive_vel * np.sin(steer_pos) * self.DT * np.sin(self.total_theta + delta_theta/2) + delta_Y_turn                       # 簡易的なオドメトリ計算によるY方向（横ずれ）の積算
        delta_Y = delta_Y_turn                       # 簡易的なオドメトリ計算によるY方向（横ずれ）の積算
        # print("tes",delta_X, delta_Y_turn, delta_Y)
        self.total_Xpos += delta_X
        self.total_Ypos += delta_Y
        # self.obs = delta_Y

        diff_angle = (self.target_steer_angle - steer_pos) / self.MAX_STEER
        return np.array([self.filtered_roll, angular_vel, drive_vel, 
                         target_drive_vel, normalized_diff_vel, steer_pos, self.total_Ypos,
                         self.target_steer_angle, diff_angle], dtype=np.float32)
    
    # バイクの傾きと位置の変化から報酬を決定
    def _reward(self, obs, action):
        # imu, angular_vel, angular_acc, action_back, body_pos_x, body_pos_y = obs
        imu, _, drive_vel, _ , _, steer_pos, _, _, _ = obs
        # print("drive_vel:", drive_vel)
        if(self.TARGET_VEL != 0):
            normalized_diff_vel = min(1, abs(self.TARGET_VEL - 0.031 * drive_vel) / abs(self.cmd_cfg["max_vel"]))  # 速度偏差の正規化: 目標速度とのズレを -1.0 ~ 1.0 の範囲に収める
        elif(self.TARGET_VEL == 0):
            normalized_diff_vel = abs(0.031 * drive_vel / self.cmd_cfg["max_vel"])
        normalized_angle = max(0.0, (np.deg2rad(45) - abs(imu)) / np.deg2rad(45))                       # 姿勢角の正規化: 45度(制限値)を1.0とし、直立(0度)に近いほど1.0、倒れるほど0.0に近づく
        normalized_torque = (abs(self.prev_torque - action[1])) / 2                             # トルク変化の正規化: 前回のトルク指令との差分。急激な出力変化（高周波な振動）へのペナルティ用
        # normalized_delta_vel = normalized_diff_vel * abs(np.cos(self.total_theta))
        normalized_delta_vel = normalized_diff_vel
        # 状態の更新（次ステップの計算用）
        self.prev_torque = action[1]
        self.prev_normalized_total_vel = self.normalized_total_vel
        self.normalized_total_vel += normalized_diff_vel                                       # 累積速度偏差の更新
        # normalized_diff_steer = abs((self.target_steer_angle - (action[0] * self.MAX_STEER)) / self.MAX_STEER)
        normalized_diff_steer = min(0.25, abs((self.target_steer_angle - steer_pos) / self.MAX_STEER)) * 4

        r_posture = self.reward_cfg["posture_unstable"] * normalized_angle**2
        r_torque = self.reward_cfg["penalty_torque_unstable"] * normalized_torque
        r_steering = self.reward_cfg["penalty_steering"] * normalized_diff_steer
        r_vel = self.reward_cfg["vel_bonus"] * (1 - normalized_delta_vel)
        self.obs_rew = normalized_diff_steer

        reward = r_posture + r_steering + r_torque + r_vel + self.reward_cfg["live_bonus"]
        reward_info = {
            "r_posture": r_posture,
            "r_steering": r_steering,
            "r_torque": r_torque,
            "r_vel": r_vel,
        }

        return reward, reward_info

    def step(self, action):
        # print(action)
        bias= 0.0021
        action_angle = action[0] * self.MAX_STEER
        action_torque = action[1] * self.MAX_TORQUE
        if self.env_cfg["real_syncro_noise"] == True:
            action_angle += np.random.normal(0, 0.005) * self.MAX_STEER  # ステアリングにノイズを加える
            action_torque += np.random.normal(0, 0.0233 * 0.002)  # トルクにノイズを加える
            if action_torque > bias:
                action_torque = action_torque-bias
            elif action_torque < -bias:
                action_torque = action_torque+bias
            else:
                action_torque = 0.0   
            self.control_queue.append(np.array([action_angle, action_torque]))
            action_angle = self.control_queue[0][0]
            action_torque = self.control_queue[0][1]

        # print(action)
        self.data.ctrl[0] = action_angle
        self.data.ctrl[1] = action_torque
        self.steering_angle = action_angle                                          # ステアリング角の計算

        for i in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            # time.sleep(0.002)
                
        obs = self._get_obs()
        reward, rew_info = self._reward(obs, action)

        terminated = bool(abs(obs[0]) > self.ANGLE_THRESHOLD 
                        #   or abs(self.total_Ypos) > self.POSY_THRESHOLD
                          )
        truncated = bool(self.step_count >= self.MAX_STEP)
        
        # 各報酬をエピソード合計に加算
        self.ep_rew_posture += rew_info["r_posture"]
        self.ep_rew_steering += rew_info["r_steering"]
        self.ep_rew_torque += rew_info["r_torque"]
        self.ep_rew_vel += rew_info["r_vel"]

        if terminated:
            reward += self.reward_cfg["penalty_if_truncated"]
        if(self.step_count - self.memory_count > 400):
            # if np.random.random() < 1/1000:
                # self.reset()
                sign = -np.sign(self.target_steer_angle)
                self.target_steer_angle = sign * np.random.uniform(self.exclude_val, self.range_max)
                # time.sleep(1)
                self.memory_count = self.step_count
                # print(self.target_steer_angle)

        self.step_count += 1
            
        info = {}
        if terminated or truncated:  
            info["ep_rew_posture"] = self.ep_rew_posture
            info["ep_rew_steering"] = self.ep_rew_steering      
            info["ep_rew_torque"] = self.ep_rew_torque     
            info["ep_rew_vel"] = self.ep_rew_vel

            # 簡潔さ重視：ゼロ除算回避と平均化を1行で処理
            info.update({
                k: (getattr(self, k) / max(1, self.step_count)) / abs(self.reward_cfg[w])
                if self.reward_cfg[w] != 0 else 0
                    for k, w in (("ep_rew_posture", "posture_unstable"), ("ep_rew_vel", "vel_bonus"),
                       ("ep_rew_steering", "penalty_steering"))
            })        

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        angle = self.env_cfg["initial_tilt_deg"]
        init_steer_angle = self.env_cfg["initial_steer_deg"] 
        if self.env_cfg["init_steer_noise"] == True:
            init_steer_angle += np.deg2rad(np.random.uniform(-45, 45))
        self.data.qpos[8] = init_steer_angle
        self.data.ctrl[0] = init_steer_angle

        if self.env_cfg["init_tilt_noise"] == True:
            angle += np.random.normal(0, np.deg2rad(self.NOISE_ANGLE))
        self.data.qpos[3:7] = [1, 0, 0, 0]
        if(self.cmd_cfg["noise_init_speed"]==True):
            init_vel = - np.random.uniform(0, self.cmd_cfg["init_noise_range"])
            self.data.qvel[0] = init_vel
            self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]] = init_vel / 0.031
            self.data.qvel[self.model.jnt_dofadr[self.f_wheel_id]] = init_vel / 0.031
        self.range_max = self.cmd_cfg["target_steer_angle_range"]
        self.exclude_val = np.deg2rad(10.0) # 除外したい角度
        # self.exclude_val = np.deg2rad(0.0) # 除外したい角度

        # 10 〜 self.range_max の範囲で乱数を生成し、ランダムに 1 か -1 を掛ける
        sign = np.random.choice([-1, 1])
        self.target_steer_angle = sign * np.random.uniform(self.exclude_val, self.range_max)
        self.TARGET_VEL = -(0.2 + 0.3*(1 - abs(self.target_steer_angle) / self.range_max))
        # self.TARGET_VEL = -self.cmd_cfg["target_vel"]
        if(self.cmd_cfg["noise_target_vel"]==True):
            self.TARGET_VEL -= (self.cmd_cfg["max_vel"] * np.random.uniform(0, self.cmd_cfg["noise_range"]))  # what is the base vel?
        if(abs(self.TARGET_VEL) < 0.02):
            self.TARGET_VEL = 0

        # print(self.TARGET_VEL)

        mujoco.mju_euler2Quat(self.data.qpos[3:7], np.array([angle, 0, 0]), "xyz")    

        self.step_count = 1
        self.memory_count = 1
        self.wheel_pos = self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]]
        self.total_odometry = 0.0
        self.prev_odometry = 0.0
        self.total_theta = 0.0
        self.total_Xpos = 0.0
        self.total_Ypos = 0.0
        self.steering_angle = 0.0
        self.normalized_total_vel = 0.0
        self.prev_action = 0.0
        self.torque = 0.0
        self.prev_torque = 0.0
        self.filtered_roll = 0.0
        self.prev_normalized_total_vel = 0.0

        self.ep_rew_posture = 0.0
        self.ep_rew_Ytotal = 0.0
        self.ep_rew_steering = 0.0
        self.ep_rew_torque = 0.0
        self.ep_rew_vel = 0.0
        self.ep_rew_total_vel = 0.0
        self.control_queue = deque([np.zeros(2)] * self.control_queue.maxlen, maxlen=self.control_queue.maxlen)
        self.encoder_queue = deque([np.zeros(2)] * self.control_queue.maxlen, maxlen=self.control_queue.maxlen)

        # for _ in range(50):
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
