import numpy as np
from typing import Tuple

class MadgwickFilter:
    """
    Madgwickフィルター (ジャイロ/加速度のみ、地磁気不使用版)
    姿勢をクォータニオンで推定します。
    """
    
    # クォータニオン [w, x, y, z] の初期値 (静止状態を仮定)
    # 初期値は通常 [1, 0, 0, 0] で、回転なしを意味します。
    # MuJoCoのクォータニオン順序 [w, x, y, z] と合わせます。
    SEq: np.ndarray
    beta: float
    deltat: float

    def __init__(self, deltat: float, gyro_meas_error: float = 5.0):
        """
        Madgwickフィルターを初期化します。

        Args:
            deltat: サンプリング周期 (秒)。MuJoCoのタイムステップ (model.opt.timestep) を使用。
            gyro_meas_error: ジャイロスコープの計測誤差 (deg/s)。
                             この値が大きいほど、加速度計による補正が強くなります。
        """
        self.deltat = deltat
        
        # ジャイロ誤差を rad/s に変換
        gyro_meas_error_rad = np.deg2rad(gyro_meas_error)
        
        # beta値を計算 (Madgwickのゲイン)
        # beta = sqrt(3/4) * gyroMeasError
        self.beta = np.sqrt(3.0 / 4.0) * gyro_meas_error_rad
        
        # 初期クォータニオン
        self.SEq = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def update(self, w_x: float, w_y: float, w_z: float, 
               a_x: float, a_y: float, a_z: float) -> np.ndarray:
        """
        フィルターの更新を実行します。

        Args:
            w_x, w_y, w_z: ジャイロスコープ測定値 (rad/s)
            a_x, a_y, a_z: 加速度計測定値 (m/s^2)
            
        Returns:
            np.ndarray: 更新されたクォータニオン [w, x, y, z]
        """
        
        q = self.SEq
        
        # --- 1. 加速度計の正規化 ---
        norm_acc = np.sqrt(a_x * a_x + a_y * a_y + a_z * a_z)
        if norm_acc == 0:
            # 加速度がゼロの場合は処理をスキップ (ジャイロ積分のみ)
            return self._integrate_gyro(q, w_x, w_y, w_z)

        a_x /= norm_acc
        a_y /= norm_acc
        a_z /= norm_acc

        # 補助変数 (Cコードのhalf/twoの変数に相当)
        half_q = 0.5 * q
        two_q = 2.0 * q

        # --- 2. 目的関数 f と勾配 (SEqHatDot) の計算 ---
        # f (重力ベクトル誤差)
        f_1 = two_q[1] * q[3] - two_q[0] * q[2] - a_x
        f_2 = two_q[0] * q[1] + two_q[2] * q[3] - a_y
        f_3 = 1.0 - two_q[1] * q[1] - two_q[2] * q[2] - a_z

        # ヤコビ行列 J の要素 (J_11or24, J_12or23, J_13or22, J_14or21)
        J_11or24 = two_q[2]
        J_12or23 = 2.0 * q[3]
        J_13or22 = two_q[0]
        J_14or21 = two_q[1]
        J_32 = 2.0 * J_14or21
        J_33 = 2.0 * J_11or24
        
        # 勾配の計算 (SEqHatDot = J^T * f)
        SEqHatDot = np.array([
            J_14or21 * f_2 - J_11or24 * f_1,
            J_12or23 * f_1 + J_13or22 * f_2 - J_32 * f_3,
            J_12or23 * f_2 - J_33 * f_3 - J_13or22 * f_1,
            J_14or21 * f_1 + J_11or24 * f_2
        ])
        
        # 勾配の正規化
        norm_grad = np.linalg.norm(SEqHatDot)
        if norm_grad == 0:
            return self._integrate_gyro(q, w_x, w_y, w_z)
            
        SEqHatDot /= norm_grad
        
        # --- 3. クォータニオン微分 (積分項) ---
        qDot_omega = self._get_qDot_omega(q, w_x, w_y, w_z)

        # --- 4. 積分 (推定されたクォータニオン微分) ---
        q_new = q + (qDot_omega - (self.beta * SEqHatDot)) * self.deltat
        
        # --- 5. 最終正規化 ---
        self.SEq = q_new / np.linalg.norm(q_new)
        
        return self.SEq

    def _integrate_gyro(self, q, w_x, w_y, w_z) -> np.ndarray:
        """ジャイロスコープのみでクォータニオンを更新する (補正なし)"""
        qDot_omega = self._get_qDot_omega(q, w_x, w_y, w_z)
        q_new = q + qDot_omega * self.deltat
        self.SEq = q_new / np.linalg.norm(q_new)
        return self.SEq

    def _get_qDot_omega(self, q, w_x, w_y, w_z) -> np.ndarray:
        """ジャイロスコープによるクォータニオン微分を計算"""
        half_q = 0.5 * q
        
        # Cコードの SEqDot_omega の計算
        return np.array([
            -half_q[1] * w_x - half_q[2] * w_y - half_q[3] * w_z,
            half_q[0] * w_x + half_q[2] * w_z - half_q[3] * w_y,
            half_q[0] * w_y - half_q[1] * w_z + half_q[3] * w_x,
            half_q[0] * w_z + half_q[1] * w_y - half_q[2] * w_x
        ])

    def get_rpy_degrees(self) -> Tuple[float, float, float]:
        """
        推定されたクォータニオンをロール, ピッチ, ヨー (度) に変換します。
        """
        q0, q1, q2, q3 = self.SEq # q = [w, x, y, z]
        
        # ロール (Roll) - x軸周り
        roll_rad = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
        
        # ピッチ (Pitch) - y軸周り
        sinp = 2 * (q0 * q2 - q3 * q1)
        # sinp = max(min(sinp, 1), -1) # 数値誤差対策
        pitch_rad = np.arcsin(sinp)
        
        # ヨー (Yaw) - z軸周り
        yaw_rad = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
        
        return np.rad2deg(roll_rad), np.rad2deg(pitch_rad), np.rad2deg(yaw_rad)