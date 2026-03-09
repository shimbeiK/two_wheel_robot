import numpy as np

class PythonConfig:
    def get_train_cfg():
        train_cfg_dict = {
            "policy": "MlpPolicy",
            "policy_kwargs": {
                "net_arch": [64, 64],
                "log_std_init": -0.5,  # 初期の行動分散を小さくして安定化
            },
            "device": "cpu",  
            "learning_rate": 3e-4,
            "n_steps": 8192,                # ← 増やすと学習安定
            "batch_size": 256,
            "gamma": 0.995,                # ← 0.99くらいで安定化
            # "n_epochs": 10,
            # "ent_coef": 0.05,             # ← 0.01くらいで探索促進
            # "gae_lambda": 0.95,
            # "max_grad_norm": 0.3,           # ← 0.5くらいで安定化
            "clip_range": 0.2,              # ← better
            "normalize_advantage": True,    # ← Trueにすべし
            "verbose": 1,
            "tensorboard_log": "tboard_logs/kourin25",
        }
        return train_cfg_dict
    
    def get_cfgs():
        env_cfg = {
            # Termination bounds (converted np.pi/6 to approx 30 degrees)
            "termination_if_roll_greater_than": np.deg2rad(45.0),
            "termination_if_posY_greater_than": 0.05, # meters
            "termination_if_step_count_greater_than": 20000, # steps
            "frame_skip": 5,  # Number of physics steps per environment step

            # Initial Base state
            "initial_tilt_deg": np.deg2rad(0),      # The 3.7 degree initialization from MuJoCo code
            "initial_steer_deg": np.deg2rad(0),   # The 3.7 degree initialization from MuJoCo code
            "initial_torque" : -0.0,                #[-1, 1]
            "init_noise": False,                    # Whether to add noise to the initial tilt angle
            "noise_angle": 1.,                     # [0, 90]deg  Initial tilt noise range in degrees (±)
            "action_noise": False,                  # Whether to add noise to the action (torque) during training
            "action_noise_range": 0.01,             # [0, 1] * max_torque or angle.  Action noise range as a fraction of max action (e.g., 0.1 for ±10% noise)
            "real_syncro_noise": True,
            
            # Action scale (ネットワーク出力 [-1, 1] をそれぞれの物理量に変換)
            "steering_angle_scale": np.deg2rad(80), # Action 0 のスケール（角度）
            "drive_torque_scale": 0.021,             # Action 1 のスケール（トルク）            
            "clip_actions": 1.0, 
        }
        
        obs_cfg = {
            "obs_scales": {
                "roll": 1.0,
                "ang_vel": 1.0,
                "ang_acc": 1.0,
            },
        }
        
        reward_cfg = {
            "penalty_if_truncated": -0.0,       # big penalty when bike roll down
            "penalty_torque_unstable": -0,       # penalty if output is unstable
            "penalty_steering": -0,                 # bonus abs steering angle is smaller
            "Ypos_penalty": -1.3,                   # bonus if Y pos is near at 0 
            "total_Xvel_penalty": -0.0,                   # bonus if real vel is simillar with target vel

            "Xvel_penalty": .7,                   # bonus if real vel is simillar with target vel
            "posture_unstable": 1.0,             # bonus if posture is stable
        }
        
        # Commands are not strictly needed for stationary balancing, but kept to prevent pipeline breakage
        cmd_cfg = {
            "num_commands": 3, 
            "max_vel": 1.0, # m/s. 15cm/s.
            "target_vel": 0.1,
            "noise": True,
            "noise_range": 0.5,   #[0, 1]          
        }

        return env_cfg, obs_cfg, reward_cfg, cmd_cfg