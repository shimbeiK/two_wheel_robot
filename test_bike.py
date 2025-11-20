'''
必要なこと
 ービジュアライズの確認
 ー観測値の確認
 ー制御入力の確認
 ー物理環境の調整（timestepsとか。必要？）
'''
import mujoco
import mujoco.viewer
import numpy as np
import time

# XMLファイルを読み込み
MODEL_PATH = "sim_env/bike.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

dt = 0.01

print(model.nu)
for i in range(model.nu):
    data.ctrl[i] = 0.0  # 初期化

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

# シミュレーションデータの観測：IMU, 本体の位置、向き、モータの回転角位置、速度、角速度
def observe():
    # obs = {'imu': None, 'body_pos': None, 'body_arg': None, 'F_motor_pos': None, 'F_motor_arg': None, 'R_motor_pos': None, 'R_motor_arg': None}
    # # ジョイントの位置と速度
    # obs['imu'] = data.sensordata.copy()
    # obs['body_pos'] = data.qpos.copy()
    # obs['body_arg'] = data.qvel.copy()
    # obs['F_motor_pos'] = data.qpos[model.joint('F_wheel_joint').qposadr]
    # obs['F_motor_arg'] = data.qvel[model.joint('F_wheel_joint').qveladr]
    # obs['R_motor_pos'] = data.qpos[model.joint('R_wheel_joint').qposadr]
    # obs['R_motor_arg'] = data.qvel[model.joint('R_wheel_joint').qveladr]
    # return obs
    pass

# モータへの出力
def motor_controll():
    pass

# ビューアを起動
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer started. Press Ctrl+C to exit.")

    # 時間設定
    print(model.opt.timestep)
    t0 = time.time()
    t = t0
    
    # 制御ループ
    while viewer.is_running():
        while(time.time() - t < model.opt.timestep):
            mujoco.mj_step(model, data)
            viewer.sync()
            t = time.time()
            time.sleep(model.opt.timestep / 10)  # 少し待つことでCPU負荷を軽減
            data.ctrl[0] = 0
            data.ctrl[1] = 0

        # 制御入力の設定
        motor_controll()
        # 経過時間
        # print("Elapsed time:", time.time() - t0)