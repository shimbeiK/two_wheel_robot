import mujoco
import mujoco.viewer
import time
import numpy as np
import torch
import torch.nn as nn
from bike_env_v3_NonSteer import StandingEnv # ご自身の環境

# --- 1. Genesisの学習設定に合わせたActorネットワーク ---
class GenesisActor(nn.Module):
    def __init__(self, num_obs=5, num_actions=1):
        super().__init__()
        # configの actor_hidden_dims: [128, 64], activation: "elu" に対応
        self.actor = nn.Sequential(
            nn.Linear(num_obs, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.actor(x)

def load_genesis_model(pt_path, device):
    """Genesis (.pt) から重みを読み込むヘルパー関数"""
    model = GenesisActor(num_obs=5, num_actions=1).to(device)
    
    # .ptファイルのロード
    state = torch.load(pt_path, map_location=device)
    
    # rsl_rl 形式の場合、'model_state_dict' の中に重みがある
    state_dict = state.get('model_state_dict', state)
    
    # Actorの重みだけを抽出 (CriticやOptimizerの重みは無視)
    actor_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("actor."):
            # プレフィックスを調整してロード
            new_key = key.replace("actor.actor.", "actor.") # rsl_rlの仕様に合わせる微調整
            actor_state_dict[key.replace("actor.", "")] = value
    
    try:
        # strict=False で柔軟に読み込む
        model.actor.load_state_dict(actor_state_dict, strict=False)
        print("✅ モデルのロードに成功しました！")
    except Exception as e:
        print(f"⚠️ ロードエラー: {e}")
        print("ファイル内のキー:", state_dict.keys())
        
    model.eval()
    return model

# --- 2. メインのシミュレーションループ ---
def main():
    MODEL_PATH = "mjcf2/scene.xml"
    PT_MODEL_PATH = "path/to/your/genesis_model.pt" # ★Genesisの.ptパスに変更してください
    
    # 環境の初期化
    env = StandingEnv(xml_path=MODEL_PATH, render_mode="human")
    
    # PyTorchの設定
    device = torch.device("cpu")
    policy = load_genesis_model(PT_MODEL_PATH, device)
    
    obs, _ = env.reset()
    print("🚀 MuJoCoでの推論を開始します... (Ctrl+Cで停止)")
    
    with torch.no_grad():
        while True:
            # 1. 観測(obs)をPyTorchのTensorに変換
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # 2. ネットワークから行動を出力 (tanh等の出力制限がないのでそのまま)
            action_tensor = policy(obs_tensor)
            action = action_tensor.squeeze(0).cpu().numpy()
            
            # --- ⚠️ 重要なスケール調整 ---
            # Genesisのconfigにある "drive_torque_scale": 0.05 を適用する。
            # ※もし MuJoCo側の env.step() の中で既に 0.05 を掛けている場合は、
            # 以下の行は不要（コメントアウト）にしてください。
            scaled_action = action * 0.05 
            
            # 3. MuJoCo環境を1ステップ進める
            obs, reward, terminated, truncated, info = env.step(scaled_action)
            
            # print(f"Obs: {obs[0]:.3f}, Raw Action: {action[0]:.3f}, Scaled Action: {scaled_action[0]:.3f}")
            
            env.render()
            time.sleep(0.01) # 100Hz (dt=0.01) に合わせる

            if terminated or truncated:
                print("💥 転倒または終了条件に達しました。リセットします。")
                obs, _ = env.reset()

if __name__ == "__main__":
    main()