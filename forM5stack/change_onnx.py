import torch
import torch.nn as nn
from stable_baselines3 import PPO
import os

# --- M5Stack用にActor（推論部分）だけを純粋に抽出するラッパー ---
class M5StackPolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        # 1. 観測値の特徴抽出 (Flattenなど)
        features = self.policy.extract_features(observation)
        # 2. Actor(行動決定)ネットワークの隠れ層を計算
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        # 3. 最終的な行動の数値（平均値）を出力
        return self.policy.action_net(latent_pi)
# --------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "results/betters/stop_better/best_model")
device = torch.device("cpu")

# モデルの読み込み
model = PPO.load(model_path, device=device)

# カスタムラッパーを適用し、評価モードにする
onnxable_model = M5StackPolicy(model.policy)
onnxable_model.eval()

# ダミーデータの作成（観測値は9つ）
dummy_input = torch.randn(1, 9).to(device)

# ONNX形式で書き出す
torch.onnx.export(
    onnxable_model,
    dummy_input,
    "robot_brain.onnx",
    verbose=True,
    input_names=['input'],
    output_names=['action'],
    opset_version=11
)

print("ONNXモデルの書き出しが完了しました！")