#########################################################
# Construct and train the Constrained Flow Matching model
# https://github.com/fvilmos
#########################################################

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from constants import *


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for continuous flow time steps.
    Parameters:
    ----------
    channels : int
        Number of input channels

    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TemporalFiLMBlock(nn.Module):
    """
    1D Temporal Convolutional Residual Block with Feature-wise Linear Modulation (FiLM).
    """
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.film_proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        h = F.gelu(self.conv1(self.norm1(x)))
        film = self.film_proj(cond).unsqueeze(-1)
        gamma, beta = film.chunk(2, dim=1)
        h = (1.0 + gamma) * self.norm2(h) + beta
        h = self.conv2(F.gelu(h))
        return x + h

class ECA1D(nn.Module):
    """
    Efficient Channel Attention for 1D sequences
    
    Parameters:
    ----------
    k : int
        Kernel size for 1D convolution (default: 3)

    """
    def __init__(self, k=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y

class PointCloudEncoder(nn.Module):
    """
    PointCloud 1D convolutional feature encoder.
    Processes 360-degree LiDAR distance beams into a compact feature embedding.
        
    Parameters:
    ----------
    in_channels (int): input channels (1 for 1D range scan)
    embed_dim (int): output embedding dimension
    """
    def __init__(self, in_channels=1, embed_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(10),
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)


class ParkingFlowNetCFG(nn.Module):
    """
    Temporal 1D U-Net, uses dedicated Pose/Delta Encoders, and decoder Goal FiLM.
    """
    def __init__(self, traj_len=40, state_dim=6, p_uncond=0.15, hidden_dim=80, num_blocks=1):
        super().__init__()
        self.traj_len = traj_len
        self.state_dim = state_dim
        self.p_uncond = p_uncond
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.point_cloud_enc = PointCloudEncoder(in_channels=1, embed_dim=64)

        # Dedicated Ego and Goal Pose Encoders
        self.start_enc = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU()
        )
        self.goal_enc = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU()
        )
        # Relative Delta Feature Encoder (dx, dy, dist, cos_dth, sin_dth)
        self.delta_enc = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )

        self.null_pose_embed = nn.Parameter(torch.randn(1, 64) * 0.02)
        self.null_delta_embed = nn.Parameter(torch.randn(1, 32) * 0.02)

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(64),
            nn.Linear(64, 64),
            nn.GELU()
        )

        # Combined context: PointCloud (64) + Goal (64) + Start (64) + Delta (32) + Time (64) = 288
        cond_dim = 64 + 64 + 64 + 32 + 64
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Goal-focused direct decoder FiLM projection
        self.dec_goal_film = nn.Linear(64, hidden_dim * 2)

        # U-Net Architecture
        self.in_proj = nn.Conv1d(state_dim, hidden_dim, kernel_size=1)

        # Encoder Level 1: Dynamic Depth Blocks (Resolution 40 timesteps, Channels hidden_dim)
        self.enc_blocks = nn.ModuleList([TemporalFiLMBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)])
        self.down1 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1)

        # Bottleneck Level 2: Dynamic Depth Blocks (Resolution 20 timesteps, Channels hidden_dim * 2) with ECA
        self.mid1_blocks = nn.ModuleList([TemporalFiLMBlock(hidden_dim * 2, hidden_dim) for _ in range(num_blocks)])
        self.bottleneck_se = ECA1D(k=3)
        self.mid2_blocks = nn.ModuleList([TemporalFiLMBlock(hidden_dim * 2, hidden_dim) for _ in range(num_blocks)])

        # Decoder Level 1: Dynamic Depth Blocks (Resolution 40 timesteps, Channels hidden_dim) with Skip Fusion
        self.up1 = nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.dec_fuse = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.dec_blocks = nn.ModuleList([TemporalFiLMBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)])

        self.out_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, state_dim, kernel_size=1)
        )

    def load_state_dict(self, state_dict, strict=True):
        """
        Supports loading of single block and dynamic block checkpoints.
        """
        adapted_sd = {}
        for k, v in state_dict.items():
            new_k = k
            if k.startswith("enc1."):
                new_k = k.replace("enc1.", "enc_blocks.0.")
            elif k.startswith("mid1."):
                new_k = k.replace("mid1.", "mid1_blocks.0.")
            elif k.startswith("mid2."):
                new_k = k.replace("mid2.", "mid2_blocks.0.")
            elif k.startswith("dec1."):
                new_k = k.replace("dec1.", "dec_blocks.0.")
            adapted_sd[new_k] = v
        return super().load_state_dict(adapted_sd, strict=strict)

    def forward(self, tau_t, t, point_cloud_scan, rel_target_pose, cond_mask=None):
        """
        FLowNet forward function

        Parameters:
        -----------
        tau_t: path buffer 40 timesteps x 6D (B,40,6) [x, y, cos_theta, sin_theta, v, delta]
        t: time step (B,1) [t]
        point_cloud_scan: sensor point cloud(i.e. Lidar) (B,1,360) [dist0 at angle 0, ...]
        rel_target_pose: relative pose from the end goal (B,4) [dx,dy, sin_theta, cos_theta]
        cond_mask: Defaults to None, used for conditional free guidance, increase precision.
        """
        B = tau_t.size(0)

        p_start_4d = tau_t[:, 0, :4]
        p_goal_4d = rel_target_pose[..., :4]
        
        # compute deltas
        dx = p_goal_4d[:, 0:1] - p_start_4d[:, 0:1]
        dy = p_goal_4d[:, 1:2] - p_start_4d[:, 1:2]
        dist = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-6)
        c0, s0 = p_start_4d[:, 2:3], p_start_4d[:, 3:4]
        cg, sg = p_goal_4d[:, 2:3], p_goal_4d[:, 3:4]
        cos_dth = c0 * cg + s0 * sg
        sin_dth = c0 * sg - s0 * cg
        delta_feat = torch.cat([dx, dy, dist, cos_dth, sin_dth], dim=-1)

        # feed encoders with information
        e_pc = self.point_cloud_enc(point_cloud_scan)
        e_start = self.start_enc(p_start_4d)
        e_goal_cond = self.goal_enc(p_goal_4d)
        e_delta_cond = self.delta_enc(delta_feat)

        # apply conditional mask
        if cond_mask is None:
            if self.training and self.p_uncond > 0.0:
                cond_mask = (torch.rand(B, 1, device=tau_t.device) > self.p_uncond).float()
            else:
                cond_mask = torch.ones(B, 1, device=tau_t.device)

        null_goal_exp = self.null_pose_embed.expand(B, -1)
        null_delta_exp = self.null_delta_embed.expand(B, -1)
        e_goal = cond_mask * e_goal_cond + (1.0 - cond_mask) * null_goal_exp
        e_delta = cond_mask * e_delta_cond + (1.0 - cond_mask) * null_delta_exp
        e_time = self.time_emb(t.view(B, 1))

        cond_raw = torch.cat([e_pc, e_goal, e_start, e_delta, e_time], dim=-1)
        cond = self.cond_mlp(cond_raw)

        # transpose trajectory (B, T, D) -> (B, D, T)
        x = tau_t.transpose(1, 2)
        h = self.in_proj(x)

        # encoder Level 1 (Resolution 40)
        for block in self.enc_blocks:
            h = block(h, cond)
        skip = h

        # bottleneck Level 2 (Resolution 20)
        h = self.down1(h)
        for block in self.mid1_blocks:
            h = block(h, cond)
        h = self.bottleneck_se(h)
        for block in self.mid2_blocks:
            h = block(h, cond)

        # decoder Level 1 (Resolution 40) with Skip Fusion
        u = self.up1(h)
        u = self.dec_fuse(torch.cat([u, skip], dim=1))
        for block in self.dec_blocks:
            u = block(u, cond)

        # goal-focused decoder FiLM conditioning
        goal_film = self.dec_goal_film(e_goal).unsqueeze(-1)
        g_gamma, g_beta = goal_film.chunk(2, dim=1)
        u = (1.0 + g_gamma) * u + g_beta

        v_out = self.out_proj(u).transpose(1, 2)
        return v_out


class ConstrainedParkingFlowLoss(nn.Module):
    """
    Kinematic model constrained loss for non-holonomic vehicle behaviour with terminal approach tangent alignment
    Combine different losses to learn the behaviour for the final goal.
    All type of losses are wighted, final tuneing could help to improve safety
    """
    def __init__(self, stats, lambda_kin=0.2, lambda_coll=0.1, lambda_act=0.05,
                 lambda_vel=0.2, lambda_end=1.0, lambda_app=0.5):
        super().__init__()
        self.stats = stats
        self.lambda_kin = lambda_kin
        self.lambda_coll = lambda_coll
        self.lambda_act = lambda_act
        self.lambda_vel = lambda_vel
        self.lambda_end = lambda_end
        self.lambda_app = lambda_app

        angles = torch.linspace(0, 2 * torch.pi, 360)
        self.register_buffer("cos_angles", torch.cos(angles))
        self.register_buffer("sin_angles", torch.sin(angles))

    def denorm(self, tau_norm):
        """
        use statistics from the dataset to normalize values

        Parameters:
        -----------
            tau_norm: normalized trajectory
        """
        mean = self.stats['traj_mean'].to(tau_norm.device)
        std = self.stats['traj_std'].to(tau_norm.device)
        return tau_norm * std + mean

    def kinematic_loss(self, tau):
        """
        Enforces non-holonomic bicycle kinematics and zero lateral slip.
        """
        x = tau[:, :-1, 0]
        y = tau[:, :-1, 1]
        theta = torch.atan2(tau[:, :-1, 3], tau[:, :-1, 2])
        v = torch.clamp(tau[:, :-1, 4], -3.0, 3.0)
        delta = torch.clamp(tau[:, :-1, 5], -np.radians(STEERING_WHEEL_MAX_ANGLE), np.radians(STEERING_WHEEL_MAX_ANGLE))

        dx = tau[:, 1:, 0] - x
        dy = tau[:, 1:, 1] - y

        # Local lateral slip penalty
        v_lat = -dx * torch.sin(theta) + dy * torch.cos(theta)
        loss_slip = F.mse_loss(v_lat, torch.zeros_like(v_lat))

        # next state with kinematic bicycle model
        x_next = x + v * torch.cos(theta) * DT
        y_next = y + v * torch.sin(theta) * DT
        theta_next = theta + (v / WHEELBASE) * torch.tan(delta) * DT

        # combine loss for each parameter
        loss_kin = F.mse_loss(x_next, tau[:, 1:, 0]) + \
                   F.mse_loss(y_next, tau[:, 1:, 1]) + \
                   F.mse_loss(torch.cos(theta_next), tau[:, 1:, 2]) + \
                   F.mse_loss(torch.sin(theta_next), tau[:, 1:, 3])
        return (loss_kin + 2.0 * loss_slip) / (tau.size(1) - 1)

    def velocity_loss(self, tau_pred_norm, tau_target_norm):
        """
        Add a loss for velocity

        Parameters:
        ----------
            tau_pred_norm: predicted normalized path
            tau_target_norm: target normalized path
        """
        loss = F.mse_loss(tau_pred_norm[..., 4], tau_target_norm[..., 4]) + \
               F.mse_loss(tau_pred_norm[..., 5], tau_target_norm[..., 5])
        return loss / tau_pred_norm.size(1)

    def end_state_loss(self, tau_pred_norm, tau_target_norm):
        """
        Loss for end state.

        Parameters:
        -----------
            tau_pred_norm: predicted normalized path
            tau_target_norm: target normalized path

        """
        final_pred = tau_pred_norm[:, -1, :]
        final_target = tau_target_norm[:, -1, :]
        pos_err = F.mse_loss(final_pred[:, :2], final_target[:, :2])
        cos_sim = torch.clamp(final_pred[:, 2] * final_target[:, 2] +
                              final_pred[:, 3] * final_target[:, 3], -1.0, 1.0)
        heading_err = 1.0 - cos_sim
        return pos_err + heading_err.mean()

    def approach_tangent_loss(self, tau_pred, tau_target,k_app=6):
        """
        Enforces colinearity and tangent alignment during terminal approach (last 6 waypoints. k_app).
        Penalizes lateral deviation from target axis and heading discrepancy in the final meter.
        """

        app_pred = tau_pred[:, -k_app:, :]
        goal_target = tau_target[:, -1, :]

        gx = goal_target[:, 0:1]
        gy = goal_target[:, 1:2]
        g_cos = goal_target[:, 2:3]
        g_sin = goal_target[:, 3:4]

        # heading colinearity with goal orientation
        cos_sim = app_pred[:, :, 2] * g_cos + app_pred[:, :, 3] * g_sin
        loss_head = (1.0 - torch.clamp(cos_sim, -1.0, 1.0)).mean()

        # lateral error relative to goal stall axis: e_lat = -(x - gx)*sin(g) + (y - gy)*cos(g)
        dx = app_pred[:, :, 0] - gx
        dy = app_pred[:, :, 1] - gy
        e_lat = -dx * g_sin + dy * g_cos
        loss_lat = e_lat.pow(2).mean()

        # finalization loss, final velocity and steering should be zero
        loss_rest = app_pred[:, -1, 4].pow(2).mean() + app_pred[:, -1, 5].pow(2).mean()

        return loss_head + 2.0 * loss_lat + 0.5 * loss_rest

    def collision_loss(self, tau, point_cloud_scan_raw):
        """
        Trajectory and point cloud collision loss
        
        Parameters:
        ----------
            tau: path
            point_cloud_scan_raw : point cloud points

        """
        r_dist = point_cloud_scan_raw.squeeze(1)
        x_obs = (r_dist * self.cos_angles).unsqueeze(1)
        y_obs = (r_dist * self.sin_angles).unsqueeze(1)

        dx = x_obs - tau[:, :, 0:1]
        dy = y_obs - tau[:, :, 1:2]

        theta_full = torch.atan2(tau[:, :, 3], tau[:, :, 2])
        cos_t = torch.cos(-theta_full).unsqueeze(-1)
        sin_t = torch.sin(-theta_full).unsqueeze(-1)

        x_local = dx * cos_t - dy * sin_t
        y_local = dx * sin_t + dy * cos_t

        # penalty if 0.2m nearer are the points
        pen_x = F.relu((CAR_LENGTH / 2.0 + 0.2) - torch.abs(x_local))
        pen_y = F.relu((CAR_WIDTH / 2.0 + 0.2) - torch.abs(y_local))
        return (pen_x * pen_y).mean()

    def action_smoothness_loss(self, tau):
        """
        penalize fluctuations from velocity and steering wheel movements

        Prameters:
        ----------
            tau: path
        """
        v = tau[:, :-1, 4]
        delta = tau[:, :-1, 5]
        return torch.diff(v, dim=1).pow(2).mean() + torch.diff(delta, dim=1).pow(2).mean()

    def forward(self, model, tau_1_norm, point_cloud_norm, pose_norm, point_cloud_raw, sigma_min=1e-4):
        B = tau_1_norm.size(0)

        # implement flow-matching
        # base noise distribution x_0 ~ N(0, I)
        tau_0 = torch.randn_like(tau_1_norm)

        # sample continuous time t in [0, 1]
        t = torch.rand(B, 1, 1, device=tau_1_norm.device)

        # flow-matching probability Path: mu_t = t * tau_1, sigma_t = 1 - (1 - sigma_min) * t
        sigma_t = 1.0 - (1.0 - sigma_min) * t
        tau_t = t * tau_1_norm + sigma_t * tau_0

        # conditional vector field target (Optimal Transport function):
        target_velocity = tau_1_norm - (1.0 - sigma_min) * tau_0

        # predict velocity field
        pred_velocity = model(tau_t, t.squeeze(-1), point_cloud_norm, pose_norm)
        loss_fm = F.mse_loss(pred_velocity, target_velocity)

        # one-step denormalized state prediction
        tau_1_pred = tau_t + (1.0 - t) * pred_velocity
        tau_pred_denorm = self.denorm(tau_1_pred)
        tau_target_denorm = self.denorm(tau_1_norm)

        # compute losses
        loss_kin = self.kinematic_loss(tau_pred_denorm)
        loss_vel = self.velocity_loss(tau_1_pred, tau_1_norm)
        loss_end = self.end_state_loss(tau_1_pred, tau_1_norm)
        loss_app = self.approach_tangent_loss(tau_pred_denorm, tau_target_denorm)
        loss_coll = self.collision_loss(tau_pred_denorm, point_cloud_raw)
        loss_act = self.action_smoothness_loss(tau_pred_denorm)

        t_weight = (t.squeeze() >= 0.5).float().mean()
        aux_loss = t_weight * (
            self.lambda_kin * loss_kin +
            self.lambda_vel * loss_vel +
            self.lambda_end * loss_end +
            self.lambda_app * loss_app +
            self.lambda_coll * loss_coll +
            self.lambda_act * loss_act
        )

        total_loss = loss_fm + aux_loss

        return total_loss, loss_fm.item(), loss_kin.item(), loss_end.item(), loss_vel.item(), loss_act.item(), loss_app.item()


class ParkingDataset(Dataset):
    """
    Dataset loader from the prepcomputed data (pt file)
    """
    def __init__(self, data_path):
        data = torch.load(data_path, weights_only=False)
        self.stats = data['stats']
        
        # normalize values
        trajectories = data['trajectories']
        point_cloud_scans = data['point_cloud_scans']
        self.point_cloud_raw = point_cloud_scans
        
        self.trajectories = (trajectories - self.stats['traj_mean']) / self.stats['traj_std']
        self.point_cloud_scans = (point_cloud_scans - self.stats['point_cloud_mean']) / self.stats['point_cloud_std']
        
        p_mean = self.stats['pose_mean'][..., :4]
        p_std = self.stats['pose_std'][..., :4]

        rel_poses = data['rel_target_poses'][..., :4]
        self.rel_poses = (rel_poses - p_mean) / p_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return (self.trajectories[idx], self.point_cloud_scans[idx],
                self.rel_poses[idx], self.point_cloud_raw[idx])

def compute_model_complexity(model, device="cpu"):
    """
    Computes model parameters and computational complexity (FLOPs / MACs) for single-sample inference.
    """
    model.eval()
    macs = 0
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    hooks = []
    def conv_hook(m, i, o):
        nonlocal macs
        _, cout, lout = o.shape
        macs += (m.in_channels // m.groups) * cout * m.kernel_size[0] * lout

    def linear_hook(m, i, o):
        nonlocal macs
        macs += (i[0].numel() // m.in_features) * m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    dummy_tau = torch.zeros(1, model.traj_len, model.state_dim, device=device)
    dummy_t = torch.zeros(1, 1, device=device)
    dummy_pc = torch.zeros(1, 1, 360, device=device)
    dummy_pose = torch.zeros(1, 4, device=device)

    with torch.no_grad():
        model(dummy_tau, dummy_t, dummy_pc, dummy_pose)

    for h in hooks:
        h.remove()

    flops = 2 * macs
    return {
        'params': params,
        'macs': macs,
        'flops': flops,
        'mmacs': macs / 1e6,
        'mflops': flops / 1e6,
        'gflops': flops / 1e9
    }


def train(nb_epochs=EPOCS, batch_size=BATCH_SIZE, lr=1e-3,dataset_path=DATASET_NAME, save_path="parking_flownet.pth", resume_checkpoint=None, num_blocks=1):
    """
    Train the model.

    Parameters:
    -----------
        nb_epochs: number of epocs. Defaults to EPOCS.
        batch_size: Batch size. Defaults to BATCH_SIZE.
        lr: learning rate. Defaults to 1e-3.
        dataset_path: path and name of the dataset. Defaults to DATASET_NAME.
        save_path : model name and path to save. Defaults to "parking_flownet.pth".
        resume_checkpoint: reload previously traned model. Defaults to None.
        num_blocks : number of repetitive FiLM blocks. Defaults to 1.
    """
    num_threads = min(8, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} (threads={num_threads})", flush=True)
    
    # create dataloader
    dataset = ParkingDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    # create model
    model = ParkingFlowNetCFG(traj_len=TRAJ_LEN, state_dim=6, p_uncond=0.15, num_blocks=num_blocks).to(device)
    
    # resuming checkpoint
    if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        print(f"Resuming weights from {resume_checkpoint}...", flush=True)
        ckpt = torch.load(resume_checkpoint, map_location=device)
        sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(sd)
    
    # get model complexity
    complexity = compute_model_complexity(model, device=device)
    print(f"Total Model Parameters: {complexity['params']:,}", flush=True)
    print(f"Computational Complexity per Forward Pass: {complexity['mmacs']:.2f} MMACs ({complexity['mflops']:.2f} MFLOPs / {complexity['gflops']:.4f} GFLOPs)", flush=True)

    loss_fn = ConstrainedParkingFlowLoss(dataset.stats, lambda_kin=0.2, lambda_coll=0.1, lambda_act=0.05, lambda_vel=0.2, lambda_end=1.0, lambda_app=0.5).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb_epochs, eta_min=1e-5)

    start_time = time.time()

    for epoch in range(nb_epochs):
        model.train()
        total_loss_accum = 0.0
        cfm_loss_accum = 0.0
        kin_loss_accum = 0.0
        end_loss_accum = 0.0
        app_loss_accum = 0.0
        vel_loss_accum = 0.0
        act_loss_accum = 0.0

        for tau_norm, point_cloud_norm, pose_norm, point_cloud_raw in dataloader:
            tau_norm = tau_norm.to(device)
            point_cloud_norm = point_cloud_norm.to(device)
            pose_norm = pose_norm.to(device)
            point_cloud_raw = point_cloud_raw.to(device)

            # compute loss
            loss, cfm_l, kin_l, end_l, vel_l, act_l, app_l = loss_fn(model, tau_norm, point_cloud_norm, pose_norm, point_cloud_raw)

            optimizer.zero_grad()
            loss.backward()

            # gradient clipping, update optimizer
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # update loss
            total_loss_accum += loss.item()
            cfm_loss_accum += cfm_l
            kin_loss_accum += kin_l
            end_loss_accum += end_l
            app_loss_accum += app_l
            vel_loss_accum += vel_l
            act_loss_accum += act_l

        # update LR scheduler
        scheduler.step()

        n = len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == nb_epochs - 1:
            lr_current = optimizer.param_groups[0]['lr']

            # print some stats 
            print(f"Epoch [{epoch+1:3d}/{nb_epochs:3d}] | Total: {total_loss_accum/n:7.4f} | "
                  f"FM: {cfm_loss_accum/n:6.4f} | End: {end_loss_accum/n:6.4f} | App: {app_loss_accum/n:6.4f} | "
                  f"Kin: {kin_loss_accum/n:6.4f} | Vel: {vel_loss_accum/n:6.4f} | LR: {lr_current:.6f}", flush=True)

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f}s ({elapsed/nb_epochs:.2f}s/epoch)", flush=True)

    torch.save({'model_state_dict': model.state_dict(),'stats': dataset.stats,'num_blocks': num_blocks}, save_path)
    print(f"Model checkpoint successfully saved to {save_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Constrained Parking Flow Matching Model")
    parser.add_argument("--epochs", type=int, default=EPOCS, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_blocks", type=int, default=1, help="Number of repetitive FiLM blocks per stage (1, 2, 3...)")
    parser.add_argument("--data_path", type=str, default=DATASET_NAME, help="Dataset path")
    parser.add_argument("--save_path", type=str, default="parking_flownet.pth", help="Checkpoint save path")
    parser.add_argument("--resume", type=str, default=None, help="Path to existing checkpoint to resume")
    args = parser.parse_args()

    train(nb_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, dataset_path=args.data_path,
        save_path=args.save_path, resume_checkpoint=args.resume, num_blocks=args.num_blocks)