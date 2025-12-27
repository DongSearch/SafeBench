# safebench/agent/rl/sac_fm_fixed.py  (SAC v2 refactor with Flow Matching)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from fnmatch import fnmatch
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import datetime

from safebench.util.torch_util import CUDA, CPU, kaiming_init
from safebench.agent.base_policy import BasePolicy
from safebench.agent.rl.FFM import FMTrainer  # Updated Flow Matching trainer

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.fc_std = nn.Linear(256, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.min_val = 1e-3
        self.apply(kaiming_init)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.tanh(self.fc_mu(x))
        # Ensure standard deviation is positive using softplus
        std = self.softplus(self.fc_std(x)) + self.min_val
        return mu, std

class Critic(nn.Module):
    """Value Network V(s)"""
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.apply(kaiming_init)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Q(nn.Module):
    """Soft Q-Network Q(s, a)"""
    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.apply(kaiming_init)

    def forward(self, x, a):
        x = x.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((x, a), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SAC_FM(BasePolicy):
    name = 'SAC'
    type = 'offpolicy'

    def __init__(self, config, logger):
        self.logger = logger
        self.buffer_start_training = config['buffer_start_training']
        self.lr = config['lr']
        self.continue_episode = 0
        self.state_dim = config['ego_state_dim']
        self.action_dim = config['ego_action_dim']
        self.min_Val = torch.tensor(config['min_Val']).float()
        self.batch_size = config['batch_size']
        self.update_iteration = config['update_iteration']
        self.gamma = config['gamma']
        self.tau = config['tau']
        
        # Logging structure for evaluation analysis
        self.eval_logs = {
            'step': [],
            'ood_score': [],
            'sdf_score': [],
            'is_risky': [],
            'velocity': [],
            'is_controlled' : []
        }

        self.model_id = config['model_id']
        self.model_path = os.path.join(config['ROOT_DIR'], config['model_path'])
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        # Initialize Networks
        self.policy_net = CUDA(Actor(self.state_dim, self.action_dim))
        self.value_net = CUDA(Critic(self.state_dim))
        self.Q_net = CUDA(Q(self.state_dim, self.action_dim))
        self.Target_value_net = CUDA(Critic(self.state_dim))

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.Q_optimizer = optim.Adam(self.Q_net.parameters(), lr=self.lr)

        self.value_criterion = nn.MSELoss()
        self.Q_criterion = nn.MSELoss()

        # Synchronize target value network
        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        # ------------------------------------------------------------------
        # [New] Flow Matching & Safety Integration
        # ------------------------------------------------------------------
        self.fm_trainer = FMTrainer(self.state_dim, self.action_dim, lr=self.lr)
        self.ood_scores = []
        self.ood_penalty = 0.0
        self.ood_threshold = 1.0
        self.buffer_start_threshold = 300  # Minimum steps before applying OOD penalty
        self.mode = 'train'
        
        # Thresholds for Out-of-Distribution (OOD) detection
        self.ood_soft_threshold = 0
        self.ood_hard_threshold = 0

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.policy_net.train()
            self.value_net.train()
            self.Q_net.train()
        elif mode == 'eval':
            self.policy_net.eval()
            self.value_net.eval()
            self.Q_net.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def get_action(self, state, infos, deterministic=False):
        state_tensor = CUDA(torch.FloatTensor(state))
        mu, log_sigma = self.policy_net(state_tensor)

        if deterministic:
            action = mu
        else:
            sigma = torch.exp(log_sigma)
            dist = Normal(mu, sigma)
            z = dist.sample()
            action = torch.tanh(z)

        # Heuristic: Adjust OOD sensitivity based on steering angle
        angle = abs(infos[0].get('angle', 0.0))
        if angle > 0.05:
            self.ood_soft_threshold, self.ood_hard_threshold = 2.0, 10.0
        else:
            self.ood_soft_threshold, self.ood_hard_threshold = 2.5, 4.0

        current_ood, sdf_ood = 0.0, 1.0
        control_triggered = 0
        is_risky_label = 1 if infos[0].get('is_risky', False) else 0

        if self.mode == "eval":
            # Use FM Trainer to check novelty (OOD) and safety (SDF)
            current_ood, sdf_ood = self.fm_trainer.predict_ood(state, action)

            # --- Control Logic (Emergency Interventions) ---
            # 1. Safety-based Intervention (SDF)
            if 0.2 < sdf_ood < 0.5:
                # Moderate risk: Decelerate
                action[..., 0] = torch.clamp(action[..., 0] - 1.0, min=-1.0)
                control_triggered = 1
                print(f"sdf:de-acceleration | ood: {current_ood:.2f}")
            elif sdf_ood < 0.2:
                # High risk: Emergency brake
                action[..., 0] = -3.0
                control_triggered = 1
                print("sdf:emergency brake")
            else:
                # 2. Novelty-based Intervention (FM/OOD)
                # Intervention only triggered when vehicle has certain speed
                is_moving = state[0][2] > 3.0 or (len(state) == 2 and state[0][1] > 3.0)
                if is_moving:
                    if self.ood_soft_threshold < current_ood < self.ood_hard_threshold:
                        action[..., 0] = torch.clamp(action[..., 0] - 1.0, min=-1.0)
                        control_triggered = 1
                        print(f"fm:de-acceleration | ood: {current_ood:.2f}")
                    elif self.ood_hard_threshold <= current_ood:
                        action[..., 0] = -2.0
                        control_triggered = 1
                        print(f"fm:emergency brake | ood: {current_ood:.2f}")

            # Record logs for visualization
            self.eval_logs['step'].append(len(self.eval_logs['step']))
            self.eval_logs['ood_score'].append(current_ood)
            self.eval_logs['sdf_score'].append(sdf_ood)
            self.eval_logs['is_controlled'].append(control_triggered)
            self.eval_logs['is_risky'].append(is_risky_label)
            self.eval_logs['velocity'].append(state[0][0])

        return CPU(action)

    def get_action_log_prob(self, state):
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        z = dist.sample()
        action = torch.tanh(z)
        # Tanh correction for log probability
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + self.min_Val)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob, z, batch_mu, batch_log_sigma

    def train(self, replay_buffer):
        current_len = replay_buffer.buffer_len
        if current_len < self.buffer_start_training * self.update_iteration:
            return

        # Train the Flow Matching dynamics model first
        fm_loss, ood_score, _ = self.fm_trainer.train(replay_buffer)

        if ood_score is not None:
            self.ood_scores.append(np.clip(ood_score, 0.0, 5.0))

        for _ in range(self.update_iteration):
            batch = replay_buffer.sample(self.batch_size)
            bn_s = CUDA(torch.FloatTensor(batch['state']))
            bn_a = CUDA(torch.FloatTensor(batch['action']))
            bn_r = CUDA(torch.FloatTensor(batch['reward'])).unsqueeze(-1)
            bn_s_ = CUDA(torch.FloatTensor(batch['n_state']))
            bn_d = CUDA(torch.FloatTensor(1 - batch['done'])).unsqueeze(-1)

            target_value = self.Target_value_net(bn_s_)

            # [OOD Penalty] Reduces Q-target for states with high novelty (unstable)
            next_q_value = bn_r + bn_d * self.gamma * target_value - self.ood_penalty

            excepted_value = self.value_net(bn_s)
            excepted_Q = self.Q_net(bn_s, bn_a)

            sample_action, log_prob, z, batch_mu, batch_log_sigma = self.get_action_log_prob(bn_s)
            excepted_new_Q = self.Q_net(bn_s, sample_action)
            next_value = excepted_new_Q - log_prob

            # V-loss optimization
            V_loss = self.value_criterion(excepted_value, next_value.detach()).mean()
            self.value_optimizer.zero_grad()
            V_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            # Q-loss optimization
            Q_loss = self.Q_criterion(excepted_Q, next_q_value.detach()).mean()
            self.Q_optimizer.zero_grad()
            Q_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.Q_net.parameters(), 0.5)
            self.Q_optimizer.step()

            # Policy-loss (Pi) optimization
            log_policy_target = excepted_new_Q - excepted_value
            pi_loss = (log_prob * (log_prob - log_policy_target).detach()).mean()
            self.policy_optimizer.zero_grad()
            pi_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            # Soft target update
            for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)

    def save_model(self, episode):
        states = {
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'Q_net': self.Q_net.state_dict(),
            'fm_trainer': self.fm_trainer.state_dict() # Includes dynamics model & OOD stats
        }
        filepath = os.path.join(self.model_path, f'model.sac.{self.model_id}.{episode:04}.torch')
        self.logger.log(f'>> Saving {self.name} model to {filepath}')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self, episode=None):
        if episode is None:
            episode = -1
            for _, _, files in os.walk(self.model_path):
                for name in files:
                    if fnmatch(name, "*torch"):
                        cur_episode = int(name.split(".")[-2])
                        if cur_episode > episode:
                            episode = cur_episode

        filepath = os.path.join(self.model_path, f'model.sac.{self.model_id}.{episode:04}.torch')

        if os.path.isfile(filepath):
            self.logger.log(f'>> Loading {self.name} model from {filepath}')
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.value_net.load_state_dict(checkpoint['value_net'])
            self.Q_net.load_state_dict(checkpoint['Q_net'])

            # Load FM weights if they exist, otherwise initialize from scratch
            if 'fm_trainer' in checkpoint:
                self.logger.log(">> Loading FlowMatching/OOD weights...")
                self.fm_trainer.load_state_dict(checkpoint['fm_trainer'])
            else:
                self.logger.log(">> FM weights not found. Initializing FM from scratch.", color='yellow')

            self.continue_episode = episode
        else:
            self.logger.log(f'>> No {self.name} model found at {filepath}', 'red')

    def calculate_auroc(self, y_true, y_score):
        """Manual AUROC calculation using Wilcoxon-Mann-Whitney logic."""
        y_true = np.array(y_true)
        y_score = np.array(y_score)

        if len(np.unique(y_true)) < 2:
            return 0.5

        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        n_pos, n_neg = len(pos), len(neg)

        combined = np.column_stack((y_score, y_true))
        combined = combined[combined[:, 0].argsort()]

        ranks = np.arange(1, len(combined) + 1)
        pos_ranks_sum = np.sum(ranks[combined[:, 1] == 1])

        auc = (pos_ranks_sum - (n_pos * (n_pos + 1)) / 2.0) / (n_pos * n_neg)
        return auc

    def save_eval_report(self, episode_id):
        """Generates a visual plot of FM/SDF scores and intervention points."""
        if not self.eval_logs['ood_score']:
            print(">> [Report] No eval data to report.")
            return

        steps = np.array(self.eval_logs['step'])
        ood_scores = np.array(self.eval_logs['ood_score'])
        sdf_scores = np.array(self.eval_logs['sdf_score'])
        is_risky = np.array(self.eval_logs['is_risky'])
        controlled = np.array(self.eval_logs['is_controlled'])

        # Calculate performance metrics
        metrics_str = "Metrics N/A"
        if len(np.unique(is_risky)) > 1:
            ood_auroc = self.calculate_auroc(is_risky, ood_scores)
            sdf_auroc = self.calculate_auroc(is_risky, 1.0 - sdf_scores)
            metrics_str = f"OOD AUROC: {ood_auroc:.4f} | SDF AUROC: {sdf_auroc:.4f}"

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot FM Novelty Score
        ax1.set_xlabel('Step')
        ax1.set_ylabel('FM Score (Novelty)', color='tab:blue')
        l_ood, = ax1.plot(steps, ood_scores, color='tab:blue', label='FM Score', alpha=0.8)

        # Plot SDF Safety Score (on secondary Y axis)
        ax2 = ax1.twinx()
        ax2.set_ylabel('SDF Score (Safety: 1=Safe, 0=Dangerous)', color='tab:orange')
        l_sdf, = ax2.plot(steps, sdf_scores, color='tab:orange', label='SDF Score', alpha=0.8)

        legend_elements = [l_ood, l_sdf]

        # Draw red dashed lines where the system intervened
        control_indices = np.where(controlled == 1)[0]
        if len(control_indices) > 0:
            first_ctrl = True
            for idx in control_indices:
                label_str = 'Intervention' if first_ctrl else ""
                l_ctrl = ax1.axvline(x=steps[idx], color='red', linestyle='--', alpha=0.4, label=label_str)
                if first_ctrl:
                    legend_elements.append(l_ctrl)
                    first_ctrl = False

        # Highlight ground truth (GT) risk zones in red background
        risk_indices = np.where(is_risky == 1)[0]
        if len(risk_indices) > 0:
            first_risk = True
            for idx in risk_indices:
                label_str = 'Risk GT' if first_risk else ""
                l_risk = ax1.axvline(x=steps[idx], color='red', alpha=0.08, label=label_str)
                if first_risk:
                    legend_elements.append(l_risk)
                    first_risk = False

        plt.title(f"Episode {episode_id} Analysis\n{metrics_str}")
        labs = [l.get_label() for l in legend_elements]
        ax1.legend(legend_elements, labs, loc='upper left', frameon=True)

        save_path = os.path.join(self.model_path, f"report_ep{episode_id}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f">> [Report] Plot saved to {save_path}")
        self.reset_eval_log()

    def reset_eval_log(self):
        self.eval_logs = {'step': [], 'ood_score': [], 'sdf_score': [], 'is_risky': [], 'velocity': [], 'is_controlled' : []}
