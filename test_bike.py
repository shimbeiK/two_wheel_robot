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
from madgwick import MadgwickFilter
from scipy.spatial.transform import Rotation as R

# XMLファイルを読み込み
MODEL_PATH = "two_wheel_robot/sim_env/bike.xml"
MODEL_PATH = "mjcf2/scene.xml"
init_angle = -45  # フォークの初期角度設定(deg)
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 接触全体を無効化
# model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
for i, p in enumerate(model.pair_geom1):
    g1 = model.pair_geom1[i]
    g2 = model.pair_geom2[i]

    # 例: geom1="wheel", geom2="body" のペアだけ無効化
    if model.geom(g1).name == "tire_holder" and model.geom(g2).name == "tire_top":
        model.pair_contype[i] = 0    # 0 → 接触生成しない
    if model.geom(g1).name == "body_obj" and model.geom(g2).name == "tire_back":
        model.pair_contype[i] = 0    # 0 → 接触生成しない

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

dt = model.opt.timestep
a = 45
num = 0.021
madgwick_filter = MadgwickFilter(model.opt.timestep, gyro_meas_error=1.0)
print(model.opt.timestep)
print(model.nu)
for i in range(model.nu):
    data.ctrl[i] = 0.0  # 初期化

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


obs = {'imu': None, 'body_pos': None, 'body_arg': None, 'F_motor_pos': None, 'F_motor_arg': None, 'R_motor_pos': None, 'R_motor_arg': None}
# シミュレーションデータの観測：IMU, 本体の位置、向き、モータの回転角位置、速度、角速度
def observe():
    # gyro_meas_error: ジャイロスコープの計測誤差 (deg/s)
    madgwick_filter.update(
        data.sensordata[adr_gyro],
        data.sensordata[adr_gyro+1],
        data.sensordata[adr_gyro+2],
        data.sensordata[adr_acc],
        data.sensordata[adr_acc+1],
        data.sensordata[adr_acc+2]
    )
    rotmat = data.xmat[1].reshape(3, 3)
    rot = R.from_matrix(rotmat)
    angle = rot.as_euler('xyz', degrees=False)
    # obs['imu'] = madgwick_filter.get_rpy_degrees()
    obs['imu'] = np.rad2deg(angle)
    obs['body_pos'] = data.qpos.copy()
    obs['body_arg'] = data.qvel.copy()
    obs['F_motor_pos'] = data.sensordata[adr_F_pos]
    obs['F_motor_vel'] = data.sensordata[adr_F_vel]
    obs['R_motor_pos'] = data.sensordata[adr_R_pos] % (np.pi*2)  # 0〜2πに正規化  
    obs['R_motor_vel'] = data.sensordata[adr_R_vel]
    # print("obs['imu'] =", np.deg2rad(obs['imu'][0]))
    return obs

# モータへの出力
def motor_controll():
    pass

counter = 0
# ビューアを起動
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer started. Press Ctrl+C to exit.")
    # data.qpos[4] = np.deg2rad(20)
    # if(data.qpos.shape[0] == 8):
    #     data.qpos[7] = np.deg2rad(init_angle)
    # elif(data.qpos.shape[0] == 9):
    #     data.qpos[8] = np.deg2rad(init_angle)
    data.qpos[8] = np.deg2rad(init_angle)
    data.qpos[3:7] = [1, 0, 0, 0]
 
    # 2. Define the Euler rotation (roll, pitch, yaw)
    # converting 45 degrees pitch to quaternion
    mujoco.mju_euler2Quat(data.qpos[3:7], np.array([np.deg2rad(0), 0, 0]), "xyz")    
    # data.ctrl[1] = 0.2   # 後輪トルク
    mujoco.mj_forward(model, data)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    # 時間設定
    print(model.opt.timestep)
    t0 = time.time()
    t = t0
    curr_gyro=0
    prev_gyro=0
    while viewer.is_running():
        a += 1
        # print(num)
        if abs(a) >= 200:
            num = -num
            a=0

        time.sleep(0.002)
        # 1. 制御入力
        data.ctrl[0] = np.deg2rad(init_angle)    # fork の角度目標
        # data.ctrl[1] = -0.084   # 後輪トルク
        data.ctrl[2] = num   # 後輪トルク
        # data.ctrl[2] = 0.1     # 後輪トルク

        # 2. 1 ステップ進める
        mujoco.mj_step(model, data)
        obs = observe()
        if counter == 0:
            print("Initial observation:", np.rad2deg(obs['imu']))
            time.sleep(2)
        curr_gyro = data.sensor("imu_gyro").data.copy()[0]
        angular_accel_est = (curr_gyro - prev_gyro) / dt   
        # Update previous value for next loop
        prev_gyro = curr_gyro
        if(counter % 10 == 1):
            # print("dt:", dt)
            # print("diff:", curr_gyro - prev_gyro)
            # print("angular vel:", curr_gyro)
            # print("angular acc:", angular_accel_est)
            print("obs['imu'] =", obs['imu'])
            if abs(obs['imu'][0]) > 45:
                print(counter, "倒れた")
                time.sleep(10)
            body_pos_x = data.qpos.copy()[0]
            body_pos_y = data.qpos.copy()[1]

            # print(np.sqrt(body_pos_x**2 + body_pos_y**2))
            print("obs['imu'] =", obs['imu'])
            # print("obs['body_pos'] =", obs['body_pos'])
            # print("obs['body_arg'] =", obs['body_arg'])
            # print("obs['F_motor_pos'] =", np.rad2deg(obs['F_motor_pos']))
            # print("obs['F_motor_vel'] =", obs['F_motor_vel'])
            # print("obs['R_motor_pos'] =", np.rad2deg((obs['R_motor_pos']))
            # print("obs['R_motor_vel'] =", obs['R_motor_vel'])
        counter += 1
        # print(counter)

        viewer.sync()    