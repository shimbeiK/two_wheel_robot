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

# XMLファイルを読み込み
MODEL_PATH = "two_wheel_robot/sim_env/bike.xml"
MODEL_PATH = "mjcf2/scene.xml"
init_angle = -60  # フォークの初期角度設定(deg)
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

counter = 0
# ビューアを起動
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer started. Press Ctrl+C to exit.")
    data.qpos[4] = np.deg2rad(1)
    data.qpos[8] = np.deg2rad(init_angle)
    # data.ctrl[1] = 0.2   # 後輪トルク
    mujoco.mj_forward(model, data)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    while viewer.is_running():

        time.sleep(0.002)
        # 1. 制御入力
        data.ctrl[0] = np.deg2rad(init_angle)    # fork の角度目標
        # data.ctrl[1] = -0.084   # 後輪トルク
        data.ctrl[2] = -0.042   # 後輪トルク
        # data.ctrl[2] = 0.1     # 後輪トルク

        # 2. 1 ステップ進める
        mujoco.mj_step(model, data)

        viewer.sync()    