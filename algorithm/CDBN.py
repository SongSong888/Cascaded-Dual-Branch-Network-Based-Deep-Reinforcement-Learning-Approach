import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from torch.distributions import Normal
from Conf import SystemConfig


class DualBranchFeatureNet(torch.nn.Module):
    def __init__(self, output_dim, state_dim, fc1_dim, hidden_dim=64, num_heads=4):
        super(DualBranchFeatureNet, self).__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim必须能被num_heads整除"

        self.conv1 = torch.nn.Conv1d(1, 4, kernel_size=8, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=1)
        self.pooling1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.pooling2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        self.embedding_layer = torch.nn.Linear(1, hidden_dim)

        self.fc1 = torch.nn.Linear(fc1_dim, 32)

        # 第一层多头注意力
        self.query_layer1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.key_layer1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value_layer1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output_proj1 = torch.nn.Linear(hidden_dim, hidden_dim)

        # 第二层多头注意力
        self.query_layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.key_layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value_layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output_proj2 = torch.nn.Linear(hidden_dim, hidden_dim)

        # 正则化组件
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        self.dropout = torch.nn.Dropout(0.1)

        # 后续处理
        self.fc2 = torch.nn.Linear(state_dim * hidden_dim, 32)
        self.output_layer = torch.nn.Linear(64, output_dim)
        self.bn1 = torch.nn.BatchNorm1d(8)

    def _multi_head_attention(self, x, query_layer, key_layer, value_layer, output_proj, layer_norm):
        batch_size, seq_len, _ = x.shape

        # 生成Q/K/V
        Q = query_layer(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = key_layer(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = value_layer(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_weights, V)

        # 合并多头
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attended = output_proj(attended)

        # 残差连接与正则化
        x = layer_norm(x + self.dropout(attended))
        return x

    def forward(self, x):
        # 分支1: 卷积处理
        x1 = x.unsqueeze(1)
        x1 = self.pooling1(F.relu(self.conv1(x1)))
        x1 = self.pooling2(F.relu(self.conv2(x1)))
        x1 = torch.flatten(x1, start_dim=1)
        x1 = F.relu(self.fc1(x1))

        # 分支2: 双层注意力处理
        x2 = x.unsqueeze(2)
        x2 = F.relu(self.embedding_layer(x2))

        # 第一层注意力
        x2 = self._multi_head_attention(x2, self.query_layer1, self.key_layer1,
                                        self.value_layer1, self.output_proj1, self.norm1)
        # 第二层注意力
        # x2 = self._multi_head_attention(x2, self.query_layer2, self.key_layer2,
        #                                 self.value_layer2, self.output_proj2, self.dyt2)

        # 动态变换与输出
        x2 = torch.flatten(x2, start_dim=1)
        x2 = F.relu(self.fc2(x2))

        # 合并分支
        x12 = torch.cat((x1, x2), dim=1)
        return self.output_layer(x12)

    def freeze_parameters(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True

# -------------------- 特征提取网络结束 --------------------

# -------------------- 上层DQN网络开始 --------------------
class UpperDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, SystemConfig.UPPER_ACTION_SPACE)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_parameters(self):
        return list(self.fc1.parameters()) + list(self.fc2.parameters())

# -------------------- 上层DQN网络结束 --------------------

# -------------------- 下层SAC网络开始 --------------------
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(64 + SystemConfig.UPPER_ACTION_SPACE, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc_mu = torch.nn.Linear(64, SystemConfig.LOWER_ACTION_DIM)
        self.fc_std = torch.nn.Linear(64, SystemConfig.LOWER_ACTION_DIM)
        self.action_bound = 1

    def forward(self, x, deterministic=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        if deterministic:
            action = torch.tanh(mu) * self.action_bound
            return action
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        action = action * self.action_bound
        return action, log_prob

    def get_parameters(self):
        return list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.fc_mu.parameters()) + list(
            self.fc_std.parameters())


class QValueNetContinuous(torch.nn.Module):
    def __init__(self):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(64 + SystemConfig.UPPER_ACTION_SPACE + SystemConfig.LOWER_ACTION_DIM, 64)
        self.fc2 = torch.nn.Linear(64, 16)
        self.fc_out = torch.nn.Linear(16, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

    def get_parameters(self):
        return list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.fc_out.parameters())


# -------------------- 下层SAC网络结束 --------------------

# -------------------- 智能体类 --------------------
class UpperAgent(torch.nn.Module):
    def __init__(self, feature_net):
        super(UpperAgent, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_net = feature_net
        self.net = UpperDQN().to(self.device)
        self.target_net = UpperDQN().to(self.device)
        self.optimizer = optim.Adam(self.net.get_parameters(), lr=SystemConfig.DQN_LR)

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, SystemConfig.UPPER_ACTION_SPACE - 1)  # 随机选取动作
        else:
            with torch.no_grad():
                q_values = self.net(state)
                return q_values.argmax().item()

    def update(self, states, actions, rewards, next_states, dones):
        # Double DQN - 使用当前网络选择最大Q值
        with torch.no_grad():
            states_feature_values = self.feature_net(states)
            target_Q_next = self.target_net(states_feature_values)
            next_states_feature_values = self.feature_net(next_states)
            Q_next = self.net(next_states_feature_values)
            Q_max_action = torch.argmax(Q_next, dim=1, keepdim=True)
            target_q_values = rewards + (1 - dones) * SystemConfig.DQN_GAMMA * target_Q_next.gather(1, Q_max_action)

        states_feature_values = self.feature_net(states)
        q_values = self.net(states_feature_values).gather(1, actions)
        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(SystemConfig.DQN_TAU * param.data +
                                    (1 - SystemConfig.DQN_TAU) * target_param.data)
        return loss.item()


class LowerAgent(torch.nn.Module):
    def __init__(self, feature_net, feature_net_optimizer, feature_net_optimizer_scheduler, actor_lr=SystemConfig.SASAC_ACTOR_LR, critic_lr=SystemConfig.SASAC_CRITIC_LR,
                 alpha_lr=SystemConfig.SASAC_ALPHA_LR,
                 target_entropy=-2, tau=SystemConfig.SASAC_TAU, gamma=SystemConfig.SASAC_GAMMA, avg_reward_init=0.0,
                 eta=SystemConfig.ETA):
        super(LowerAgent, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_net = feature_net
        self.feature_net_optimizer = feature_net_optimizer
        self.feature_net_optimizer_scheduler = feature_net_optimizer_scheduler
        # 初始化平均奖励值
        self.avg_reward_init = avg_reward_init
        self.avg_reward = torch.tensor(avg_reward_init, dtype=torch.float32, device=self.device)
        self.eta = eta
        self.actor = PolicyNetContinuous().to(self.device)  # 策略网络
        self.critic_1 = QValueNetContinuous().to(self.device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous().to(self.device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous().to(self.device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous().to(self.device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.get_parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.get_parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.get_parameters(), lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(SystemConfig.SASAC_ALPHA), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        action = self.actor(state)[0]
        return action.detach().squeeze().cpu().numpy()

    def select_test_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state, deterministic=True)
        return action.detach().squeeze().cpu().numpy()

    def calc_feature_value(self, states, upper_action):
        upper_action = upper_action.long()
        upper_action_onehot = F.one_hot(upper_action, num_classes=SystemConfig.UPPER_ACTION_SPACE).float()
        if states.device != upper_action_onehot.device:
            upper_action_onehot = upper_action_onehot.to(states.device)
        combined_feature = torch.cat([self.feature_net(states), upper_action_onehot], dim=1)
        return combined_feature

    def calc_target(self, upper_action, rewards, next_states, dones):  # 计算目标Q值
        # 计算下一个状态的动作和熵
        next_states_feature_value = self.calc_feature_value(next_states, upper_action)
        next_actions, log_prob = self.actor(next_states_feature_value)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states_feature_value, next_actions)
        q2_value = self.target_critic_2(next_states_feature_value, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, states, upper_action, lower_actions, rewards, next_states, dones):
        # 计算中心化奖励
        centered_reward_batch = rewards - self.avg_reward
        # 更新两个Q网络
        td_target = self.calc_target(upper_action, centered_reward_batch, next_states, dones)
        # 计算当前状态的Q值
        with torch.no_grad():
            states_feature_values = self.calc_feature_value(states, upper_action)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states_feature_values, lower_actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states_feature_values, lower_actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        states_feature_values = self.calc_feature_value(states, upper_action)
        critic_1_loss1 = torch.mean(F.mse_loss(self.critic_1(states_feature_values, lower_actions), td_target.detach()))
        critic_2_loss2 = torch.mean(F.mse_loss(self.critic_2(states_feature_values, lower_actions), td_target.detach()))
        feature_loss = 0.5 * (critic_1_loss1 + critic_2_loss2)
        self.feature_net_optimizer.zero_grad()
        feature_loss.backward()
        self.feature_net_optimizer.step()
        self.feature_net_optimizer_scheduler.step()

        # 更新策略网络
        with torch.no_grad():
            states_feature_values = self.calc_feature_value(states, upper_action)
        new_actions, log_prob = self.actor(states_feature_values)
        entropy = -log_prob
        q1_value = self.critic_1(states_feature_values, new_actions)
        q2_value = self.critic_2(states_feature_values, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # 更新平均奖励
        delta_r_bar = (rewards.mean() - self.avg_reward)
        self.avg_reward += self.eta * delta_r_bar.detach()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        return actor_loss.item(), critic_1_loss.item(), critic_2_loss.item()


class CDBN(torch.nn.Module):
    def __init__(self):
        super(CDBN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dual_branch_feature_net = DualBranchFeatureNet(64, SystemConfig.UPPER_STATE_DIM, 48).to(self.device)
        self.feature_net_optimizer = torch.optim.Adam(self.dual_branch_feature_net.parameters(), lr=SystemConfig.FEATURE_NET_LR)
        self.feature_net_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.feature_net_optimizer, milestones=[288 * 10, 288 * 30, 288 * 70], gamma=0.1)
        self.upper_agent = UpperAgent(self.dual_branch_feature_net)
        self.lower_agent = LowerAgent(self.dual_branch_feature_net, self.feature_net_optimizer, self.feature_net_optimizer_scheduler)
        self.buffer = deque(maxlen=SystemConfig.BUFFER_SIZE)

    def update_upper_agent(self, batch_buffer):
        states, actions, _, rewards, _, next_states, dones = zip(*batch_buffer)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        upper_loss = self.upper_agent.update(states, actions, rewards, next_states, dones)
        return upper_loss

    def update_lower_agent(self, batch_buffer):
        states, upper_action, lower_actions, _, rewards, next_states, dones = zip(*batch_buffer)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        upper_action = torch.tensor(np.array(upper_action), dtype=torch.float32).to(self.device)
        lower_actions = torch.tensor(np.array(lower_actions), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        lower_loss = self.lower_agent.update(states, upper_action, lower_actions, rewards, next_states, dones)
        return lower_loss

    def update(self):
        batch_buffer = random.sample(self.buffer, SystemConfig.BATCH_SIZE)
        upper_loss = self.update_upper_agent(batch_buffer)
        lower_loss = self.update_lower_agent(batch_buffer)
        return upper_loss, lower_loss

    def select_actions(self, states, epsilon=0.1):
        states = torch.FloatTensor(np.array(states)).to(self.device).unsqueeze(0)
        feature = self.dual_branch_feature_net(states)
        upper_action = self.upper_agent.select_action(feature, epsilon)
        upper_action_onehot = F.one_hot(torch.tensor([upper_action]),
                                        num_classes=SystemConfig.UPPER_ACTION_SPACE).float().squeeze()
        if feature.device != upper_action_onehot.device:
            upper_action_onehot = upper_action_onehot.to(feature.device).unsqueeze(0)

        combined_feature = torch.cat([feature, upper_action_onehot], dim=1)
        lower_action = self.lower_agent.select_action(combined_feature)
        return upper_action, upper_action_onehot, lower_action

    def train_state(self):
        self.dual_branch_feature_net.train()
        self.upper_agent.train()
        self.lower_agent.train()

    def test_state(self):
        self.dual_branch_feature_net.eval()
        self.upper_agent.eval()
        self.lower_agent.eval()

    def save_model(self, filepath):
        checkpoint = {
            'dual_branch_feature_net': self.dual_branch_feature_net.state_dict(),
            'feature_net_optimizer': self.feature_net_optimizer.state_dict(),
            'feature_net_optimizer_scheduler': self.feature_net_optimizer_scheduler.state_dict(),
            'upper_agent': self.upper_agent.state_dict(),
            'lower_agent': self.lower_agent.state_dict()
        }
        # 保存模型
        torch.save(checkpoint, os.path.join(filepath, f'trained_model.h5'))
        print(f"模型已保存至: {filepath}")

    def load_model(self, filepath):
        # 加载模型
        checkpoint = torch.load(os.path.join(filepath, f'trained_model.h5'), map_location=self.device)

        self.dual_branch_feature_net.load_state_dict(checkpoint['dual_branch_feature_net'])
        self.feature_net_optimizer.load_state_dict(checkpoint['feature_net_optimizer'])
        self.feature_net_optimizer_scheduler.load_state_dict(checkpoint['feature_net_optimizer_scheduler'])
        self.upper_agent.load_state_dict(checkpoint['upper_agent'])
        self.lower_agent.load_state_dict(checkpoint['lower_agent'])

        self.dual_branch_feature_net.to(self.device)

        print(f"模型已从: {filepath} 加载")
