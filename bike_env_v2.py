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
        self.frame_skip = 1
        self.MAX_STEP = max_step
        self.step_count = 0
        self.l_wheel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "tire_back_pitch")
        self.prev_angular_vel = 0.0
        self.prev_action = [0.0, 0.0]
        self.wheel_pos = self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]]
        # self.data.qpos[8] = np.deg2rad(0)
        self.data.qpos[8] = np.deg2rad(-60)


        self.ANGLE_THRESHOLD = np.pi/6  # radians
        self.POS_THRESHOLD = 0.8      # meters
        # self.x_threshold = 0.8     # meters

        # 1. render_mode を保存する
        self.render_mode = render_mode

        # 2. レンダラーとビューワーは初期値 None (使う時に作成する "Lazy initialization")
        self.viewer = None
        self.renderer = None        
        self.MAX_TORQUE = 0.042  # 最大トルク
        self.MAX_ANGLE = 30 * (np.pi / 180)
        self.MAX_STEER = 60 * (np.pi / 180)
        self.NOISE_ANGLE = 1
        # action_high = np.array([self.MAX_ANGLE, self.MAX_TORQUE], dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        # self.action_space = spaces.Box(-self.MAX_TORQUE, self.MAX_TORQUE, dtype=np.float32)

        # 観測空間：位置、速度、角度、角速度
        high = np.array([self.ANGLE_THRESHOLD, np.finfo(np.float32).max, 
                         np.finfo(np.float32).max, 1, 1, 1, 1], dtype=np.float32)
                        #  np.finfo(np.float32).max, self.MAX_ANGLE, self.MAX_TORQUE], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # 1. センサーID
        self.id_F_pos = self.model.sensor(name="F_wheel_pos").id
        self.id_F_vel = self.model.sensor(name="F_wheel_vel").id
        self.id_R_pos = self.model.sensor(name="R_wheel_pos").id
        self.id_R_vel = self.model.sensor(name="R_wheel_vel").id
        self.id_acc   = self.model.sensor(name="imu").id
        self.id_gyro  = self.model.sensor(name="imu_gyro").id

        # 2. アドレス（sensordata の中の開始 index）
        self.adr_F_pos = self.model.sensor_adr[self.id_F_pos]
        self.adr_F_vel = self.model.sensor_adr[self.id_F_vel]
        self.adr_R_pos = self.model.sensor_adr[self.id_R_pos]
        self.adr_R_vel = self.model.sensor_adr[self.id_R_vel]
        self.adr_acc   = self.model.sensor_adr[self.id_acc]
        self.adr_gyro  = self.model.sensor_adr[self.id_gyro]
        # self.madgwick_filter = MadgwickFilter(model.opt.timestep * self.frame_skip, gyro_meas_error=1.0)
        self.madgwick_filter = MadgwickFilter(self.model.opt.timestep, gyro_meas_error=1.0)

    # センサから観測する。位置はMujoco環境から得る
    def _get_obs(self, action_steer, action_back, prev_action_steer, prev_action_back):
    # gyro_meas_error: ジャイロスコープの計測誤差 (deg/s)
        # self.madgwick_filter.update(
        #     self.data.sensordata[self.adr_gyro],
        #     self.data.sensordata[self.adr_gyro+1],
        #     self.data.sensordata[self.adr_gyro+2],
        #     self.data.sensordata[self.adr_acc],
        #     self.data.sensordata[self.adr_acc+1],
        #     self.data.sensordata[self.adr_acc+2]
        # )
        rotmat = self.data.xmat[1].reshape(3, 3)
        rot = R.from_matrix(rotmat)
        angle = rot.as_euler('xyz', degrees=False)
        imu = np.rad2deg(angle)  # Convert to radians
        # imu = self.madgwick_filter.get_rpy_degrees()  # 0〜2πに正規化 
        # angular_vel = self.data.sensor("imu_gyro").data.copy()[0]+np.random.normal(0, 0.01)  # ジャイロのx軸の角速度にノイズを加える
        angular_vel = self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]]
        angular_acc = (angular_vel - self.prev_angular_vel) / (self.model.opt.timestep * self.frame_skip)   
        # Update previous value for next loop
        self.prev_angular_vel = angular_vel
        # Ac_motor_vel = self.data.sensordata[self.adr_F_vel]  # 0〜2πに正規化 
        # print(np.deg2rad(imu[0])) 
        return np.array([np.deg2rad(imu[0]), angular_vel, angular_acc, action_steer, 
                         action_back, prev_action_steer, prev_action_back], dtype=np.float32)

    # バイクの傾きと位置の変化から報酬を決定
    def _reward(self, obs):
        imu, angular_vel, angular_acc, action_steer, action_back, prev_action_steer, prev_action_back = obs
        reward = 0.0
        # """ 
        # reduce reward when the torque direction is opposite to the lean direction
        # if np.sign(imu + np.deg2rad(4.0)) != np.sign(action):
        #     reward -= 1.0 * (abs(abs((imu+np.deg2rad(4.0))/self.ANGLE_THRESHOLD) 
        #                   + abs(action/self.MAX_TORQUE)))

        # increase reward when the angle of lean is minimized
        reward += 4 * (np.deg2rad(45) - abs(imu)) / np.deg2rad(45)
        # body_pos_x = 10*self.data.qpos.copy()[0]
        # body_pos_y = 10*self.data.qpos.copy()[1]
        # # print("pos:", np.sqrt(body_pos_x**2 + body_pos_y**2))
        # reward-= 3 * np.sqrt(body_pos_x**2 + body_pos_y**2)

        # reduce reward when position is differ from center
        # reward -= 0.5 * min(2, abs(angular_vel))

        # reward += 0.5 * (1 - action_steer) # ステアリングの使用を抑制
        # reward -= 1 * abs(action_back)

        # reward += -0.4 * max(5, abs(action_steer - prev_action_steer))  # 急激なステアリング変化を抑制
        # reward -= 0.5 * (action_back - prev_action_back)**2      # 急激な後輪トルク変化を抑制

        return reward
        # """   
        # return self.step_count / 100.0

    def step(self, action):
        # print(action)
        action_angle = action[0]*self.MAX_STEER
        action_torque = action[1]*self.MAX_TORQUE
        # print(action)
        self.data.ctrl[0] = action_angle
        self.data.ctrl[1] = action_torque
        for i in range(self.frame_skip):
            # self.data.ctrl[0] = action_angle
            # self.data.ctrl[1] = action_torque
            mujoco.mj_step(self.model, self.data)
            # time.sleep(0.002)
                
        obs = self._get_obs(action[0], action[1], self.prev_action[0], self.prev_action[1])
        self.prev_action = [action[0], action[1]]
        reward = self._reward(obs)
        # if(self.step_count == 1):
        #     print("imu =", np.rad2deg(obs[0]))

        terminated = bool(abs(obs[0]) > self.ANGLE_THRESHOLD 
                        #   or abs(obs[1]) > 10
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
        # self.madgwick_filter = MadgwickFilter(self.model.opt.timestep * self.frame_skip, gyro_meas_error=1.0)
        for _ in range(50):
            mujoco.mj_forward(self.model, self.data)
        # self.data.qpos[8] = np.deg2rad(0)
        self.data.qpos[8] = np.deg2rad(-60)
        # angle = np.random.uniform(-self.NOISE_ANGLE + 3.5, self.NOISE_ANGLE + 3.5)
        angle = 4
        self.data.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mju_euler2Quat(self.data.qpos[3:7], np.array([np.deg2rad(angle), 0, 0]), "xyz")    

        self.step_count = 0
        self.wheel_pos = self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]]
        self.prev_angular_vel = 0.0
        self.prev_action = [0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(0, 0, 0.0, 0.0), {}

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
