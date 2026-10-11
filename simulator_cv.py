##########################################
# Parking with Flow-Matching
# https://github.com/fvilmos
##########################################

import os
import torch
import numpy as np
import cv2
from train import ParkingFlowNetCFG
from utils import *
from constants import *

# scale (pixels per meter), for visualization
PIXELS_PER_METER = int(640/40) # +/- 20m
GRID_SIZE_METERS = 2    # 1x1 meter grid

# Calculate grid spacing in pixels
grid_spacing = int(GRID_SIZE_METERS * PIXELS_PER_METER)

# Calculate grid spacing in pixels
grid_spacing = int(GRID_SIZE_METERS * PIXELS_PER_METER)


@torch.no_grad()
def generate_parking_path(model, point_cloud_norm, pose_norm, steps=20, cfg_scale=1.5,
                              solver='heun', start_norm=None, goal_norm=None, pin_start=True, pin_goal=True):
    """
    Samples coarse waypoints using ParkingFlowNetCFG but just the unconditional part
    Expected to produce more accurate paths, with less 'creativity', and 2x faster then CFG
    """
    model.eval()
    B = point_cloud_norm.size(0)
    device = point_cloud_norm.device

    # random trajectory
    tau = torch.randn(B, model.traj_len, 6, device=device)
    if pin_start and start_norm is not None:
        start_expanded = start_norm.to(device).expand(B, 1, -1) if start_norm.dim() == 2 else start_norm.to(device).view(1, 1, -1).expand(B, 1, -1)
        tau[:, 0:1, :] = start_expanded
    if pin_goal and goal_norm is not None:
        goal_expanded = goal_norm.to(device).expand(B, 1, -1) if goal_norm.dim() == 2 else goal_norm.to(device).view(1, 1, -1).expand(B, 1, -1)
        tau[:, -1:, :] = goal_expanded

    # prepare model input values
    dt = 1.0 / steps
    cond_mask = torch.ones(B, 1, device=device)

    # generate values using unconditional flow guidance
    for step in range(steps):
        t_val = step * dt
        t_single = torch.full((B, 1), t_val, device=device)
        v_cond = model(tau, t_single, point_cloud_norm, pose_norm, cond_mask=cond_mask)

        if pin_start and start_norm is not None:
            v_cond[:, 0, :] = 0.0
        if pin_goal and goal_norm is not None:
            v_cond[:, -1, :] = 0.0

        # Implement Heun and Euler ODE solvers 
        if solver.lower() == 'heun' and step < steps - 1:
            tau_pred = tau + v_cond * dt
            if pin_start and start_norm is not None:
                tau_pred[:, 0:1, :] = start_expanded
            
            if pin_goal and goal_norm is not None:
                tau_pred[:, -1:, :] = goal_expanded
            
            t_half = torch.full((B, 1), t_val + 0.5 * dt, device=device)
            v_cond_half = model(tau_pred, t_half, point_cloud_norm, pose_norm, cond_mask=cond_mask)

            
            if pin_start and start_norm is not None:
                v_cond_half[:, 0, :] = 0.0
            
            if pin_goal and goal_norm is not None:
                v_cond_half[:, -1, :] = 0.0
            tau = tau + v_cond_half * dt
        else:
            tau = tau + v_cond * dt

        if pin_start and start_norm is not None:
            tau[:, 0:1, :] = start_expanded
        if pin_goal and goal_norm is not None:
            tau[:, -1:, :] = goal_expanded

    return tau

@torch.no_grad()
def generate_parking_path_cfg(model, point_cloud_norm, pose_norm, steps=20, cfg_scale=1.5,
                              solver='heun', start_norm=None, goal_norm=None, pin_start=True, pin_goal=True):
    """
    Samples coarse waypoints using ParkingFlowNetCFG with Classifier-Free Guidance
    """
    model.eval()
    B = point_cloud_norm.size(0)
    device = point_cloud_norm.device

    # random trajectory
    tau = torch.randn(B, model.traj_len, 6, device=device)
    if pin_start and start_norm is not None:
        start_expanded = start_norm.to(device).expand(B, 1, -1) if start_norm.dim() == 2 else start_norm.to(device).view(1, 1, -1).expand(B, 1, -1)
        tau[:, 0:1, :] = start_expanded
    if pin_goal and goal_norm is not None:
        goal_expanded = goal_norm.to(device).expand(B, 1, -1) if goal_norm.dim() == 2 else goal_norm.to(device).view(1, 1, -1).expand(B, 1, -1)
        tau[:, -1:, :] = goal_expanded

    # prepare model input values
    dt = 1.0 / steps
    point_cloud_double = torch.cat([point_cloud_norm, point_cloud_norm], dim=0)
    pose_double = torch.cat([pose_norm, pose_norm], dim=0)
    cond_mask = torch.cat([torch.ones(B, 1, device=device), torch.zeros(B, 1, device=device)], dim=0)

    # generate values using conditional flow guidance
    for step in range(steps):
        t_val = step * dt
        t_single = torch.full((B, 1), t_val, device=device)
        t_double = torch.cat([t_single, t_single], dim=0)
        tau_double = torch.cat([tau, tau], dim=0)
        v_double = model(tau_double, t_double, point_cloud_double, pose_double, cond_mask=cond_mask)
        v_cond, v_uncond = v_double.chunk(2, dim=0)
        
        # applay CFG
        v_guided_1 = v_uncond + cfg_scale * (v_cond - v_uncond)

        if pin_start and start_norm is not None:
            v_guided_1[:, 0, :] = 0.0
        if pin_goal and goal_norm is not None:
            v_guided_1[:, -1, :] = 0.0

        # Implement Heun and Euler ODE solvers 
        if solver.lower() == 'heun' and step < steps - 1:
            tau_pred = tau + v_guided_1 * dt
            if pin_start and start_norm is not None:
                tau_pred[:, 0:1, :] = start_expanded
            
            if pin_goal and goal_norm is not None:
                tau_pred[:, -1:, :] = goal_expanded
            
            t_half = torch.full((B, 1), t_val + 0.5 * dt, device=device)
            t_double_half = torch.cat([t_half, t_half], dim=0)
            tau_double_half = torch.cat([tau_pred, tau_pred], dim=0)
            v_double_half = model(tau_double_half, t_double_half, point_cloud_double, pose_double, cond_mask=cond_mask)
            v_cond_half, v_uncond_half = v_double_half.chunk(2, dim=0)
            v_guided_2 = v_uncond_half + cfg_scale * (v_cond_half - v_uncond_half)
            
            if pin_start and start_norm is not None:
                v_guided_2[:, 0, :] = 0.0
            
            if pin_goal and goal_norm is not None:
                v_guided_2[:, -1, :] = 0.0
            tau = tau + v_guided_2 * dt
        else:
            tau = tau + v_guided_1 * dt

        if pin_start and start_norm is not None:
            tau[:, 0:1, :] = start_expanded
        if pin_goal and goal_norm is not None:
            tau[:, -1:, :] = goal_expanded

    return tau

def render_frame_cv(img, frame_idx, total_frames, global_waypoints, coarse_global, raw_flownet_x, raw_flownet_y,
                    goal_pose, boxes, lidar, angles_rad, img_size=800, map_min=MAP_MIN, map_max=MAP_MAX):
    """
    Render single frame to OpenCV image.
    """
    
    # Background
    img[:] = 255
    
    # Map bounds (thick black border)
    px_min, py_min = world_to_pixel(map_min, map_min, map_min, map_max, img_size)
    px_max, py_max = world_to_pixel(map_max, map_max, map_min, map_max, img_size)
    cv2.rectangle(img, (px_min, py_min), (px_max, py_max), (0, 0, 0), 3)
    
    # vertical lines
    for x in range(0, img_size, grid_spacing):
        cv2.line(img, (x, 0), (x, img_size), (200, 200, 200), 1)  # gray line
        
        # Add label
        meters = x / PIXELS_PER_METER
        cv2.putText(img, f"{meters:.1f}m", (x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (200, 200, 200), 1)

    # horizontal lines
    for y in range(0, img_size, grid_spacing):
        cv2.line(img, (0, y), (img_size, y), (200, 200, 200), 1)  # gray line
        # Add label
        meters = y / PIXELS_PER_METER
        cv2.putText(img, f"{meters:.1f}m", (5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (200, 200, 200), 1)

    # Obstacles
    for cx, cy, w, h in boxes:
        px, py = world_to_pixel(cx, cy, map_min, map_max, img_size)
        pw = int(w / (map_max - map_min) * img_size)
        ph = int(h / (map_max - map_min) * img_size)
        cv2.rectangle(img, (px - pw//2, py - ph//2), (px + pw//2, py + ph//2), (100, 100, 100), -1)
    
    # Goal pose ( arrow + rectangle)
    gx, gy, gyaw = goal_pose
    gp_x, gp_y = world_to_pixel(gx, gy, map_min, map_max, img_size)
    goal_car_w = int(CAR_DIMS[0] / (map_max - map_min) * img_size)  # width (longitudinal)
    goal_car_h = int(CAR_DIMS[1] / (map_max - map_min) * img_size)  # height (lateral)
    draw_rect_rotated(img, gp_x, gp_y, goal_car_w, goal_car_h, gyaw, (0, 255, 0), thickness=2)
    draw_arrow(img, gp_x, gp_y, gyaw, length=30, color=(0, 255, 0), thickness=2)
    
    # Planned CFM path
    if len(coarse_global) > 1:
        path_pts = np.array([[world_to_pixel(wp[0], wp[1], map_min, map_max, img_size) for wp in coarse_global]], dtype=np.int32)
        cv2.polylines(img, path_pts, False, (0, 255, 255), 2)
    
    # Raw FlowNet predicted points (small magenta circles)
    for fx, fy in zip(raw_flownet_x, raw_flownet_y):
        fp_x, fp_y = world_to_pixel(fx, fy, map_min, map_max, img_size)
        cv2.circle(img, (fp_x, fp_y), 2, (255, 0, 255), -1)
    
    # Current state
    curr_x, curr_y, curr_th, curr_v, curr_delta = global_waypoints[frame_idx]
    
    # LiDAR scan (blue points)
    scan = lidar.get_scan((curr_x, curr_y, curr_th), boxes)
    valid = (scan > 0.05) & (scan < MAX_POINT_CLOUD_RANGE)
    for i in np.where(valid)[0]:
        px_scan = curr_x + scan[i] * np.cos(angles_rad[i] + curr_th)
        py_scan = curr_y + scan[i] * np.sin(angles_rad[i] + curr_th)
        sp_x, sp_y = world_to_pixel(px_scan, py_scan, map_min, map_max, img_size)
        cv2.circle(img, (sp_x, sp_y), 2, (255, 0, 0), -1)
    
    # Driven path (blue line)
    if frame_idx > 0:
        path_pts = np.array([[world_to_pixel(pt[0], pt[1], map_min, map_max, img_size) for pt in global_waypoints[:frame_idx + 1]]], dtype=np.int32)
        cv2.polylines(img, path_pts, False, (255, 0, 0), 2)
    
    # Current vehicle (box + heading)
    cp_x, cp_y = world_to_pixel(curr_x, curr_y, map_min, map_max, img_size)
    car_w = int(CAR_DIMS[0] / (map_max - map_min) * img_size)  # width (longitudinal)
    car_h = int(CAR_DIMS[1] / (map_max - map_min) * img_size)  # height (lateral)
    draw_rect_rotated(img, cp_x, cp_y, car_w, car_h, curr_th, (255, 0, 0), thickness=2)
    draw_arrow(img, cp_x, cp_y, curr_th, length=30, color=(0, 0, 255), thickness=2)
    
    # Text info (top-left)
    gear_str = "FWD" if curr_v >= 0 else "REV"
    text = f"Frame {frame_idx+1}/{total_frames} | Gear: {gear_str} | Vel: {curr_v:.2f} m/s | Steer: {np.degrees(curr_delta):.1f}deg"
    cv2.putText(img, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    return img


def run_simulation_cv(dataset_path="parking_dataset_tst.pt", model_path="parking_flownet.pth", 
                      output_video="parking_simulation_cv.mp4", img_size=800, fps=12, use_cfg=False):
    """
    Run parking simulation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.isfile(dataset_path):
        dataset_path = "parking_dataset.pt"
    
    print(f"Load Dataset: {dataset_path}")
    dataset = torch.load(dataset_path, weights_only=False)
    
    test_idx = np.random.randint(0, len(dataset['trajectories']) - 1)
    test_idx = 160
    print ("total samples:",len(dataset['trajectories']), "test_idx:", test_idx)
    
    start_pose = dataset['start_poses'][test_idx]
    goal_pose = dataset['goal_poses'][test_idx]
    boxes = dataset['boxes'][test_idx]
    
    # point cloud information
    point_cloud_raw = dataset['point_cloud_scans'][test_idx:test_idx + 1]

    # relative distance between START to GOAL
    pose_raw = dataset['rel_target_poses'][test_idx:test_idx + 1]
    stats = dataset['stats']
    
    # load state dict, fix it for 1/ multiple blocks
    checkpoint = torch.load(model_path, map_location=device)
    sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    num_blocks = checkpoint.get("num_blocks", 1) if isinstance(checkpoint, dict) else 1
    max_enc_idx = 0
    for k in sd.keys():
        if k.startswith("enc_blocks."):
            idx = int(k.split(".")[1])
            max_enc_idx = max(max_enc_idx, idx + 1)
    if max_enc_idx > 0:
        num_blocks = max_enc_idx
    
    # Create the model
    model = ParkingFlowNetCFG(traj_len=40, state_dim=6, num_blocks=num_blocks).to(device)
    model.load_state_dict(sd, strict=False)
    
    # generate relative (delta) pose from START to GOAL
    point_cloud_norm = ((point_cloud_raw - stats['point_cloud_mean']) / stats['point_cloud_std']).to(device)
    pose_raw_4d = pose_raw[..., :4]
    p_mean_4d = stats['pose_mean'][..., :4]
    p_std_4d = stats['pose_std'][..., :4]
    pose_norm = ((pose_raw_4d - p_mean_4d) / p_std_4d).to(device)
    
    # CFM Waypoint Sampling
    start_canonical = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=device)
    start_norm = (start_canonical - stats['traj_mean'].squeeze().to(device)) / stats['traj_std'].squeeze().to(device)
    goal_6d = torch.cat([pose_raw_4d.to(device), torch.zeros(1, 2, device=device)], dim=-1)
    goal_norm = (goal_6d - stats['traj_mean'].squeeze().to(device)) / stats['traj_std'].squeeze().to(device)
    
    if use_cfg == True:
        tau_gen_norm = generate_parking_path_cfg(model, point_cloud_norm, pose_norm, steps=10, cfg_scale=1.5,
                                             solver='heun', start_norm=start_norm, goal_norm=goal_norm, pin_start=True, pin_goal=True)
    else:
        tau_gen_norm = generate_parking_path(model, point_cloud_norm, pose_norm, steps=10, cfg_scale=1.5,
                                              solver='heun', start_norm=start_norm, goal_norm=goal_norm, pin_start=True, pin_goal=True)
    
    tau_gen_raw = (tau_gen_norm * stats['traj_std'].to(device) + stats['traj_mean'].to(device)).cpu().squeeze(0).numpy()
    
    # Raw predicted FlowNet points in global coordinate frame
    x0, y0, th0 = start_pose
    raw_flownet_x = []
    raw_flownet_y = []
    for point in tau_gen_raw:
        lx, ly = point[0], point[1]
        gx = x0 + lx * np.cos(th0) - ly * np.sin(th0)
        gy = y0 + lx * np.sin(th0) + ly * np.cos(th0)
        raw_flownet_x.append(gx)
        raw_flownet_y.append(gy)
    
    # Local to Global Coordinate Conversion
    coarse_global = []
    for point in tau_gen_raw:
        lx, ly, c_th, s_th, v, delta = point
        gx = x0 + lx * np.cos(th0) - ly * np.sin(th0)
        gy = y0 + lx * np.sin(th0) + ly * np.cos(th0)
        gth = normalize_angle(th0 + np.arctan2(s_th, c_th))
        coarse_global.append((gx, gy, gth, v, delta))
    
    # Kinematic Bicycle Closed-Loop Simulation
    print("Simulating kinematic bicycle following...")
    global_waypoints = simulate_pure_pursuit_follow(coarse_global, sim_dt=0.02, goal_override=goal_pose)
    
    # Frame subsampling
    frame_skip = max(1, len(global_waypoints) // 300)
    render_indices = list(range(0, len(global_waypoints), frame_skip))
    if render_indices[-1] != len(global_waypoints) - 1:
        render_indices.append(len(global_waypoints) - 1)
    
    # OpenCV video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (img_size, img_size))
    
    # LiDAR simulator
    lidar = PointCloudSimulator(num_beams=360, max_range=MAX_POINT_CLOUD_RANGE)
    angles_rad = np.radians(np.linspace(0, 360, lidar.num_beams, endpoint=False))
    
    print(f"Rendering {len(render_indices)} frames...")
    for idx, frame_num in enumerate(render_indices):
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        img = render_frame_cv(img, frame_num, len(global_waypoints), global_waypoints, coarse_global,
                              raw_flownet_x, raw_flownet_y, goal_pose, boxes, lidar, angles_rad,
                              img_size=img_size, map_min=MAP_MIN, map_max=MAP_MAX)
        out.write(img)
        
        if (idx + 1) % 50 == 0:
            print(f"  Rendered {idx + 1}/{len(render_indices)} frames")
    
    out.release()
    print(f"Simulation successfully saved to {output_video}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate parking simulations")
    parser.add_argument("--nr_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--dataset", type=str, default="parking_dataset_tst.pt", help="Dataset path")
    parser.add_argument("--model", type=str, default="parking_flownet.pth", help="Model path")
    args = parser.parse_args()
    
    for i in range(args.nr_samples):
        output_video = f"parking_simulation_cv_{i:03d}.mp4" if args.nr_samples > 1 else "parking_simulation_cv.mp4"
        print(f"\n[{i+1}/{args.nr_samples}] Generating {output_video}...")
        run_simulation_cv(dataset_path=args.dataset, model_path=args.model, 
                          output_video=output_video, img_size=320, fps=30)


