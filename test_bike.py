'''
必要なこと
 ービジュアライズの確認
 ー観測値の確認
 ー制御入力の確認
 ー物理環境の調整（timestepsとか。必要？）
 入力の角度上限がないのか、観測が正しいか、観測値にノイズや遅れは生じているか
'''
import mujoco
import mujoco.viewer
import numpy as np
import time

# XMLファイルを読み込み
MODEL_PATH = "sim_env/bike.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 1. センサーID
id_F_pos = model.sensor(name="F_wheel_pos").id
id_F_vel = model.sensor(name="F_wheel_vel").id
id_R_pos = model.sensor(name="R_wheel_pos").id
id_R_vel = model.sensor(name="R_wheel_vel").id
id_acc   = model.sensor(name="imu").id
id_gyro  = model.sensor(name="imu_gyro").id

# 2. アドレス（sensordata の中の開始 index）
adr_F_pos = model.sensor_adr[id_F_pos]
adr_F_vel = model.sensor_adr[id_F_vel]
adr_R_pos = model.sensor_adr[id_R_pos]
adr_R_vel = model.sensor_adr[id_R_vel]
adr_acc   = model.sensor_adr[id_acc]
adr_gyro  = model.sensor_adr[id_gyro]

dt = 0.01
a = -80
print(model.nu)
for i in range(model.nu):
    data.ctrl[i] = 0.0  # 初期化

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


obs = {'imu': None, 'body_pos': None, 'body_arg': None, 'F_motor_pos': None, 'F_motor_arg': None, 'R_motor_pos': None, 'R_motor_arg': None}
# シミュレーションデータの観測：IMU, 本体の位置、向き、モータの回転角位置、速度、角速度
def observe():
    # 加速度計は3成分
    acc = data.sensordata[adr_acc:adr_acc+3]

    # ジョイントの位置と速度
    obs['imu'] = data.sensordata[adr_gyro:adr_gyro+3]
    obs['body_pos'] = data.qpos.copy()
    obs['body_arg'] = data.qvel.copy()
    obs['F_motor_pos'] = data.sensordata[adr_F_pos]
    obs['F_motor_vel'] = data.sensordata[adr_F_vel]
    obs['R_motor_pos'] = data.sensordata[adr_R_pos] % 360
    obs['R_motor_vel'] = data.sensordata[adr_R_vel]
    return obs

# モータへの出力
def motor_controll():
    pass

counter = 0
# ビューアを起動
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer started. Press Ctrl+C to exit.")

    # 時間設定
    print(model.opt.timestep)
    t0 = time.time()
    t = t0
    while viewer.is_running():

        # 1. 制御入力
        data.ctrl[0] = np.rad2deg(30)     # fork の角度目標
        data.ctrl[1] = 0.0     # 後輪トルク

        # 2. 1 ステップ進める
        mujoco.mj_step(model, data)

        if(counter % 1000 == 0):
            obs = observe()
            print("obs['imu'] =", obs['imu'])
            print("obs['body_pos'] =", obs['body_pos'])
            print("obs['body_arg'] =", obs['body_arg'])
            print("obs['F_motor_pos'] =", obs['F_motor_pos'])
            print("obs['F_motor_vel'] =", obs['F_motor_vel'])
            print("obs['R_motor_pos'] =", obs['R_motor_pos'])
            print("obs['R_motor_vel'] =", obs['R_motor_vel'])
        counter += 1

        viewer.sync()    
    # # 制御ループ
    # while viewer.is_running():
    #     a += 5
    #     print(a)
    #     # 一定時間シミュレーションを進める
    #     while(time.time() - t < model.opt.timestep * 100):
    #         mujoco.mj_step(model, data)
    #         viewer.sync()
    #         time.sleep(model.opt.timestep / 10)  # 少し待つことでCPU負荷を軽減

    #     data.ctrl[0] = np.deg2rad(a)
    #     data.ctrl[1] = 0.021
    #     # 制御入力の設定
    #     motor_controll()
    #     obs = observe()
    #     print("obs['imu'] =", obs['imu'])
    #     print("obs['body_pos'] =", obs['body_pos'])
    #     print("obs['body_arg'] =", obs['body_arg'])
    #     print("obs['F_motor_pos'] =", obs['F_motor_pos'])
    #     print("obs['F_motor_vel'] =", obs['F_motor_vel'])
    #     print("obs['R_motor_pos'] =", obs['R_motor_pos'])
    #     print("obs['R_motor_vel'] =", obs['R_motor_vel'])

    #     t = time.time()
    #     # 経過時間
    #     # print("Elapsed time:", time.time() - t0)