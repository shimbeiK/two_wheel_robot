'''
初動トルクが小さすぎないか？
'''

import gymnasium as gym
from gymnasium import spaces
import mujoco, json
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

    def __init__(self, model, data, render_mode=None, max_step=parameters["max_step"]):
        # super(StandingEnv(), self).__init__(gym.env)
        self.model = model
        self.data = data
        self.frame_skip = 0
        self.max_step = max_step
        self.step_count = 0

        self.angle_threshold = np.pi/4  # radians
        self.pos_threshold = 0.8      # meters
        # self.x_threshold = 0.8     # meters

        # 1. render_mode を保存する
        self.render_mode = render_mode

        # 2. レンダラーとビューワーは初期値 None (使う時に作成する "Lazy initialization")
        self.viewer = None
        self.renderer = None        
        self.RENDING = 0
        self.MAX_TRQUE = 0.042  # 最大トルク
        self.counter = 0
        self.max_angle = 90 * (np.pi / 180)
        action_high = np.array([self.max_angle, self.MAX_TRQUE], dtype=np.float32)
        # self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.action_space = spaces.Box(-self.MAX_TRQUE, self.MAX_TRQUE, dtype=np.float32)

        # 観測空間：位置、速度、角度、角速度
        high = np.array([self.angle_threshold, self.pos_threshold,
                         self.pos_threshold, np.finfo(np.float32).max], dtype=np.float32)
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
        self.madgwick_filter = MadgwickFilter(model.opt.timestep, gyro_meas_error=1.0)

    # センサから観測する。位置はMujoco環境から得る
    def _get_obs(self):
    # gyro_meas_error: ジャイロスコープの計測誤差 (deg/s)
        self.madgwick_filter.update(
            self.data.sensordata[self.adr_gyro],
            self.data.sensordata[self.adr_gyro+1],
            self.data.sensordata[self.adr_gyro+2],
            self.data.sensordata[self.adr_acc],
            self.data.sensordata[self.adr_acc+1],
            self.data.sensordata[self.adr_acc+2]
        )
        imu = self.madgwick_filter.get_rpy_degrees()  # 0〜2πに正規化 
        body_pos_x = self.data.qpos.copy()[0]
        body_pos_y = self.data.qpos.copy()[1]
        Ac_motor_vel = self.data.sensordata[self.adr_F_vel]  # 0〜2πに正規化 
        # print(np.deg2rad(imu[0])) 
        return np.array([np.deg2rad(imu[0]), body_pos_x, body_pos_y, Ac_motor_vel], dtype=np.float32)

    # バイクの傾きと位置の変化から報酬を決定
    def _reward(self, obs, action):
        imu, body_pos_x, body_pos_y, Ac_motor_vel = obs
        reward = 10 - 10*abs(imu) / self.angle_threshold -0* np.sqrt(body_pos_x**2 + body_pos_y**2) / self.pos_threshold
        # reward = reward - 2 * (np.sign(imu) != np.sign(action))

        return max(0, reward / 10)

    def step(self, action):
        # print(action)
        # target_angle = action[0]
        target_torque = action
        # for i in range(self.frame_skip):
        #     self.data.ctrl[0] = np.deg2rad(-45)
        #     self.data.ctrl[1] = target_torque
        #     mujoco.mj_step(self.model, self.data)
        self.data.ctrl[0] = np.deg2rad(60)
        self.data.ctrl[2] = target_torque
        mujoco.mj_step(self.model, self.data)
                
        self.step_count += 1
        obs = self._get_obs()
        reward = self._reward(obs, target_torque)

        done = bool(
            # abs(np.linalg.norm(self.data.xpos[1, :2])) > self.pos_threshold
            abs(obs[0]) > self.angle_threshold or
            np.sqrt(obs[1]**2 + obs[2]**2) > self.pos_threshold or
            self.step_count >= self.max_step
        )
        # 辞書の中にデータを入れる
        info = {
            "mj_data": self.data, 
            "other_info": 123
        }

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # self.madgwick_filter = MadgwickFilter(self.model.opt.timestep * self.frame_skip, gyro_meas_error=1.0)
        for _ in range(50):
            self.madgwick_filter = MadgwickFilter(self.model.opt.timestep, gyro_meas_error=1.0)
            mujoco.mj_forward(self.model, self.data)
        self.data.qpos[8] = np.deg2rad(-60)
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        self.data.qpos[4] = np.deg2rad(1)

        self.step_count = 0
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