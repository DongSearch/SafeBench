import os
import torch
import torch.nn as nn
import numpy as np
from safebench.util.torch_util import CUDA, kaiming_init


class FlowMatching(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        # dynamics_dim represents the driving-related dimensions (e.g., 3D)
        self.dynamics_dim = state_dim - 1  
        self.input_norm = nn.LayerNorm(state_dim + action_dim)

        # Vector Field Network: Predicts the velocity field for driving dynamics
        self.vf_network = nn.Sequential(
            nn.Linear(state_dim + action_dim + self.dynamics_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.dynamics_dim)
        )

        # SDF Network: Predicts safety/risk (1D output) based on state-action context
        self.sdf_network = nn.Sequential(
            nn.Linear(state_dim - 2 + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            # Narrowing the layer before the final risk output
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
        self.apply(kaiming_init)

    def forward(self, x_t, t, s, a):
        """
        x_t: [B, 3] Interpolated dynamics state
        t  : [B, 1] Time step
        s  : [B, 4] Full state
        a  : [B, action_dim] Action
        """
        sa_combined = torch.cat([s, a], dim=-1)
        sa_norm = self.input_norm(sa_combined)

        # FM Input: Concatenate current noisy state, normalized state-action, and time
        input_fm = torch.cat([x_t, sa_norm, t], dim=-1)
        v = self.vf_network(input_fm)

        # SDF calculation (focused on specific state slices, e.g., pedestrian/obstacle info)
        s_sdf = sa_norm[:, 2:]
        sdf = self.sdf_network(s_sdf)
        return v, sdf


class FMTrainer:
    def __init__(self, state_dim, action_dim, lr=1e-3, device=None):
        self.dynamics_dim = state_dim - 1
        self.model = CUDA(FlowMatching(state_dim, action_dim))
        self.device = next(self.model.parameters()).device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.sdf_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        # Training hyperparameters
        self.buffer_start_training = 100
        self.update_iteration = 3
        self.batch_size = 128
        self.sdf_alpha = 1.0  # Weight for SDF loss
        self.scaling = 10.0    # Scaling factor for target velocity
        self.momentum = 0.01   # Momentum for OOD running statistics
        self.eps = 1e-8

        # Running statistics for OOD (Out-of-Distribution) detection
        self.running_mean = torch.zeros(self.dynamics_dim, device=self.device)
        self.running_var = torch.ones(self.dynamics_dim, device=self.device)

    # --- Save/Load functions for model weights and running statistics ---
    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "running_mean": self.running_mean,
            "running_var": self.running_var
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])

        # Dimension correction for loaded stats (e.g., if switching between 4D and 3D)
        loaded_mean = state["running_mean"].to(self.device)
        loaded_var = state["running_var"].to(self.device)

        if loaded_mean.shape[0] != self.dynamics_dim:
            print(f"[FM Fix] Adjusting running stats from {loaded_mean.shape[0]} to {self.dynamics_dim}")
            self.running_mean = loaded_mean[:self.dynamics_dim]
            self.running_var = loaded_var[:self.dynamics_dim]
        else:
            self.running_mean = loaded_mean
            self.running_var = loaded_var

    def update_running_stats(self, r):
        """Updates moving average of residuals for OOD scoring."""
        if r.numel() == 0: return
        b_m = r.mean(dim=0)
        b_v = r.var(dim=0, unbiased=False)
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * b_m
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * b_v

    def get_ood_score(self, r):
        """Calculates Z-score based OOD novelty using the running stats."""
        if r.numel() == 0: return torch.zeros(0, device=self.device)
        std = torch.sqrt(torch.clamp(self.running_var, min=self.eps))
        z = (r - self.running_mean) / (std + self.eps)
        return torch.norm(z, dim=-1)

    def predict_ood(self, state, action):
        """Inference method to get safety score (SDF) and novelty score (OOD)."""
        if isinstance(state, np.ndarray): state = torch.FloatTensor(state).to(self.device)
        if isinstance(action, np.ndarray): action = torch.FloatTensor(action).to(self.device)
        if state.dim() == 1: state = state.unsqueeze(0)
        if action.dim() == 1: action = action.unsqueeze(0)

        with torch.no_grad():
            B = state.shape[0]
            # Evaluation at middle time step t=0.5
            t = torch.full((B, 1), 0.5, device=self.device)
            x_t = state[:, :self.dynamics_dim]
            v_pred, s_pred = self.model(x_t, t, state, action)
            
            s_ood = torch.sigmoid(s_pred) # Safety score (0: unsafe, 1: safe)
            f_ood = self.get_ood_score(v_pred) # Novelty score
            return f_ood.mean().item(), s_ood.mean().item()

    def train(self, buffer):
        if buffer.buffer_len < self.buffer_start_training * self.update_iteration:
            return None, None, None

        losses, oods = [], []
        for _ in range(self.update_iteration):
            batch = buffer.sample(self.batch_size)
            bn_s = CUDA(torch.FloatTensor(batch['state']))
            bn_a = CUDA(torch.FloatTensor(batch['action']))
            bn_s_ = CUDA(torch.FloatTensor(batch['n_state']))
            bn_done = CUDA(torch.FloatTensor(batch.get('done', np.zeros(len(batch['state']))))).unsqueeze(-1)

            # Separate dynamics components (e.g., pos/vel) and observation components
            bn_s_dyn = bn_s[:, :self.dynamics_dim]
            bn_s_obs = bn_s[:, self.dynamics_dim:]
            bn_s_dyn_next = bn_s_[:, :self.dynamics_dim]

            B = bn_s.shape[0]
            t = CUDA(torch.rand(B, 1))

            # Flow Matching Training: Learning the mapping from s_t to s_{t+1}
            x_t = t * bn_s_dyn_next + (1 - t) * bn_s_dyn
            v_pred, sdf_pred = self.model(x_t, t, bn_s, bn_a)
            v_target = (bn_s_dyn_next - bn_s_dyn) * self.scaling

            # Compute FM loss only for non-crash scenarios
            non_crash_mask = (bn_done < 0.5).float()
            fm_loss_raw = ((v_pred - v_target) ** 2).mean(dim=1, keepdim=True)
            fm_loss = (fm_loss_raw * non_crash_mask).sum() / (non_crash_mask.sum() + 1e-6)

            # SDF Training: Identifying unsafe zones (obstacles or collisions)
            # is_unsafe is True if observation indicates danger or done flag is high
            is_unsafe = (bn_s_obs > 0.5) | (bn_done > 0.5)
            sdf_target = torch.where(is_unsafe, torch.zeros_like(sdf_pred), torch.ones_like(sdf_pred))
            
            # Use high weight for unsafe cases to handle class imbalance
            pos_weight = torch.where(is_unsafe, 1000.0, 1.0)
            sdf_loss = (self.sdf_loss_fn(sdf_pred, sdf_target) * pos_weight).mean()

            total_loss = fm_loss + self.sdf_alpha * sdf_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # OOD calculation and running stats update
            with torch.no_grad():
                res = (v_pred - v_target).detach()
                valid_indices = torch.nonzero(non_crash_mask.squeeze(), as_tuple=True)[0]
                if len(valid_indices) > 0:
                    self.update_running_stats(res[valid_indices])
                    batch_ood = self.get_ood_score(res)
                    ood_mean = batch_ood[valid_indices].mean().item()
                else:
                    ood_mean = 0.0

            losses.append(total_loss.item())
            oods.append(ood_mean)
            print(f"[FM-Separated] fm_loss={fm_loss.item():.6f}, sdf_loss={sdf_loss.item():.6f}, ood={ood_mean:.2f}")

        return total_loss.item(), ood_mean, None
