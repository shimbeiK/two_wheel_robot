import gymnasium as gym
import numpy as np
import time,random, json, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

# dqn_lib.py内での絶対パス取得
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
            "max_step": json_param["max_step"],
            "per_alpha": json_param["per_alpha"],
            "per_beta": json_param["per_beta"],
            "per_beta_increment": json_param["per_beta_increment"]
}


# Q値を出力するためのニューラルネットワーク
class N_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(parameters["input_size"], 128) # 入力4次元 → 隠れ層128ユニットへの全結合層（例：観測が4次元のとき）
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, parameters["action_size"])

    # この関数を実行すれば深層学習が出来る
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
    
class ReplayBuffer:
    # バッファーの定義，バッチサイズの代入
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size) # dequeはリストのようなもの

    #関数によっては__len__()という関数が内部で知らないうちに使われる)
    def __len__(self):
        return len(self.buffer)

    # フィードバック[カートの位置，カートの速度，ポールの角度，ポールの角速度]を取得しリプレイバッファに保存．
    # 溢れたら古いデータを削除．ここでの経験は[状態，行動，報酬，次の状態]の4つ
    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done) # 経験をタプルに格納
        self.buffer.append(data)

    # バッファーからランダムにバッチサイズ分のデータを取得
    def sampling(self):
        datas = random.sample(self.buffer, self.batch_size)
        
        # データを分割してそれぞれの変数に格納
        # print(datas)
        state = torch.FloatTensor(np.stack([x[0] for x in datas]))
        action = torch.LongTensor(np.array([x[1] for x in datas]).astype(np.int64))
        reward = torch.FloatTensor(np.array([x[2] for x in datas]).astype(np.float32))
        next_state = torch.FloatTensor(np.stack([x[3] for x in datas]))
        done = torch.FloatTensor(np.array([x[4] for x in datas]).astype(np.int32))
        # print(state, action)
        return state, action, reward, next_state, done
    
# --- PER用のSumTree実装 ---
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.count = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.count < self.capacity:
            self.count += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

# --- Prioritized Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.alpha = parameters["per_alpha"] # 優先度の度合い
        self.beta = parameters["per_beta"]   # 補正の度合い
        self.beta_increment = parameters["per_beta_increment"]
        self.epsilon = 0.01 # 優先度が0にならないようにするための小さな値

    def __len__(self):
        return self.tree.count

    def add(self, state, action, reward, next_state, done):
        # 新しい経験には最大の優先度を与える（少なくとも一度は再生されるように）
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0
        data = (state, action, reward, next_state, done)
        self.tree.add(max_p, data)

    def sampling(self):
        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.count * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        # データの整形
        state = torch.FloatTensor(np.stack([x[0] for x in batch]))
        action = torch.LongTensor(np.array([x[1] for x in batch]).astype(np.int64))
        reward = torch.FloatTensor(np.array([x[2] for x in batch]).astype(np.float32))
        next_state = torch.FloatTensor(np.stack([x[3] for x in batch]))
        done = torch.FloatTensor(np.array([x[4] for x in batch]).astype(np.int32))
        
        is_weights = torch.FloatTensor(is_weights)

        return state, action, reward, next_state, done, idxs, is_weights

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (error + self.epsilon) ** self.alpha
            self.tree.update(idx, p)