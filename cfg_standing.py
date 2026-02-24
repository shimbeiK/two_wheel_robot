import numpy as np

class PythonConfig:
    def get_train_cfg():
        train_cfg_dict = {
            "policy": "MlpPolicy",
            "policy_kwargs": {
                "net_arch": [64, 64],
                "log_std_init": 0.3,  # 初期の行動分散を小さくして安定化
            },
            "device": "cpu",  
            "learning_rate": 3e-4,
            "n_steps": 8192,                # ← 増やすと学習安定
            "batch_size": 256,
            "gamma": 0.997,
            # "n_epochs": 10,
            # "ent_coef": 0.05,             # ← 0.01くらいで探索促進
            # "gae_lambda": 0.95,
            # "max_grad_norm": 0.3,           # ← 0.5くらいで安定化
            "clip_range": 0.3,              # ← better
            "normalize_advantage": True,    # ← Trueにすべし
            "verbose": 1,
            "tensorboard_log": "tboard_logs/kourin25",
        }
        return train_cfg_dict
    
    def get_cfgs():
        env_cfg = {
            # Termination bounds (converted np.pi/6 to approx 30 degrees)
            "termination_if_roll_greater_than": np.deg2rad(45.0),
            "termination_if_posX_greater_than": 0.5, # meters
            "termination_if_posY_greater_than": 0.5, # meters
            "termination_if_step_count_greater_than": 700, # steps
            "frame_skip": 1,  # Number of physics steps per environment step
            
            # Initial Base state
            "initial_tilt_deg": np.deg2rad(4), # The 3.7 degree initialization from MuJoCo code
            "initial_steer_deg": np.deg2rad(-60), # The 3.7 degree initialization from MuJoCo code
            "initial_torque" : 0.0021,
            "init_noise": False,  # Whether to add noise to the initial tilt angle
            "noise_angle": 0.5,    # Initial tilt noise range in degrees (±)
            "action_noise": False, # Whether to add noise to the action (torque) during training
            "action_noise_range": 0.01, # Action noise range as a fraction of max action (e.g., 0.1 for ±10% noise)
            # Action scale (ネットワーク出力 [-1, 1] をそれぞれの物理量に変換)
            "steering_angle_scale": np.deg2rad(60), # Action 0 のスケール（角度）
            "drive_torque_scale": 0.021,              # Action 1 のスケール（トルク）            
            "clip_actions": 1.0, 
        }
        
        obs_cfg = {
            "num_obs": 7, # Roll angle (rad), Angular Velocity (rad/s), Angular Acceleration (rad/s^2)
            "obs_scales": {
                "roll": 1.0,
                "ang_vel": 1.0,
                "ang_acc": 1.0,
            },
        }
        
        reward_cfg = {
            "survival_bonus": 0.0,      # base reward
            "upright_posture": 10.0,     # Matches: * 1
            "angular_vel_penalty": 7.0, # Matches: * 1
            "pos_penalty": 1.0,           # Matches:  * 1
            "steering_change_penalty": - 1, # Matches:  * 1
            "torque_change_penalty": - 0,   # Matches: * 1
            "reward_if_truncated": 30.0, # Matches: * 1
        }
        
        # Commands are not strictly needed for stationary balancing, but kept to prevent pipeline breakage
        command_cfg = {
            "num_commands": 0, 
        }

        return env_cfg, obs_cfg, reward_cfg, command_cfg