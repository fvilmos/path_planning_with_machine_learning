##########################################
# Generate dataset in a single .pt file
# https://github.com/fvilmos
##########################################
import random
import torch
import numpy as np
import rsplan
from utils import *
from constants import *


def generate_reeds_shepp_trajectory(start_pose, goal_pose, traj_len=TRAJ_LEN, turning_radius=5.0, step=0.1):
    """
    Generates a kinematically valid Reeds-Shepp trajectory using rsplan with smooth velocity profiling.
        
    Parameters:
    ----------
    Returns [traj_len, 6]: (x, y, cos_theta, sin_theta, v, delta) in the local frame
    """
    x0, y0, theta0 = start_pose
    xg, yg, thetag = goal_pose

    try:
        path = rsplan.planner.path(tuple(start_pose), tuple(goal_pose), turning_radius, 0.0, step)
        
        # extract waypoints, [[x,y,yaw, direction, curvature]]
        wps = np.array([[w.x, w.y, w.yaw, w.driving_direction, w.curvature] for w in path.waypoints()], dtype=np.float64)
    except Exception:
        # Fallback, straight-line interpolation with smooth ramp down to 0 at goal
        dx_g = (xg - x0) * np.cos(-theta0) - (yg - y0) * np.sin(-theta0)
        dy_g = (xg - x0) * np.sin(-theta0) + (yg - y0) * np.cos(-theta0)
        dtheta_g = (thetag - theta0 + np.pi) % (2 * np.pi) - np.pi
        
        # lookahead for N steps, constant velocity
        states = []
        for k in range(traj_len):
            t = k / (traj_len - 1)
            th = t * dtheta_g
            v_k = MAX_VELOCITY * np.sin(np.pi * t)
            states.append([t * dx_g, t * dy_g, np.cos(th), np.sin(th), float(v_k), 0.0])
        
        return np.array(states, dtype=np.float32)

    if len(wps) < 2:
        return generate_reeds_shepp_trajectory(start_pose, goal_pose, traj_len)

    # Transform global waypoints into the local start frame
    cos0, sin0 = np.cos(theta0), np.sin(theta0)
    local_x = (wps[:, 0] - x0) * cos0 + (wps[:, 1] - y0) * sin0
    local_y = -(wps[:, 0] - x0) * sin0 + (wps[:, 1] - y0) * cos0
    local_th = (wps[:, 2] - theta0 + np.pi) % (2 * np.pi) - np.pi

    # Smooth velocity profile
    v_profile = compute_smooth_velocity_profile(wps, max_v=MAX_VELOCITY, a_max=0.8, a_lat_max=0.5)

    # Build state vectors and resample to traj_len uniform arc-length steps
    states_full = []
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(local_x), np.diff(local_y)))))
    for k in range(len(wps)):
        v = v_profile[k]
		# wps format: [x, y, yaw, direction, curvature]
        delta = extract_steering(wps[k, 4])
        if k == len(wps) - 1:
            delta = 0.0  # Center steering at parking final rest

        # constract the state
        states_full.append([
            float(local_x[k]),
            float(local_y[k]),
            float(np.cos(local_th[k])),
            float(np.sin(local_th[k])),
            float(v),
            float(delta)
        ])
    states_full = np.array(states_full, dtype=np.float32)

    # check arc, is small curvature, interpolate a line
    total_arc = arc[-1]
    if total_arc < 0.01:
        return states_full[:1].repeat(traj_len, axis=0)

    # echidistant sampling 
    sample_arc = np.linspace(0.0, total_arc, traj_len)

    # interpolate values: (x, y, cos_theta, sin_theta, v, delta)
    out = np.zeros((traj_len, 6), dtype=np.float32)
    out[:, 0] = np.interp(sample_arc, arc, states_full[:, 0])
    out[:, 1] = np.interp(sample_arc, arc, states_full[:, 1])
    out[:, 4] = np.interp(sample_arc, arc, states_full[:, 4])
    out[:, 5] = np.interp(sample_arc, arc, states_full[:, 5])

    # interpolate heading
    th_unwrapped = np.unwrap(local_th)
    sample_th = np.interp(sample_arc, arc, th_unwrapped)
    out[:, 2] = np.cos(sample_th)
    out[:, 3] = np.sin(sample_th)

    return out

def create_parking_dataset(num_samples=1500, save_path="parking_dataset.pt"):
    """
    Generate dataset
        
    Parameters:
    ----------
    num_samples (int, optional): number of samples to be generated. Defaults to 1500.
    save_path (str, optional): Name of the dataset file. Defaults to "parking_dataset.pt".
    """
    print(f"Generating {num_samples} parking scenarios...")
    
    trajectories = []
    point_cloud_scans = []
    rel_target_poses = []
    start_poses = []
    goal_poses = []
    obstacle_lists = []
    
    ray_angles = np.linspace(0, 2 * np.pi, 360, endpoint=False, dtype=np.float32)
    
    # try to get trajecotries without collisions
    attempts = 0
    while len(trajectories) < num_samples and attempts < num_samples * NR_TRIES_TRAJECTORY_GEN:
        attempts += 1

        # randomize starting point
        start_x = random.uniform(-3.0, 3.0)
        start_y = random.uniform(-3.0, 3.0)
        start_theta = random.uniform(-np.pi, np.pi)

        # randomize GOAL to lean the recovery
        if random.random() < 0.5:
            # reference-style far goals
            goal_x = random.choice([
                random.uniform(MAP_MIN + 8, MAP_MIN + 15),
                random.uniform(MAP_MAX - 15, MAP_MAX - 8)
            ])
            goal_y = random.choice([
                random.uniform(MAP_MIN + 8, MAP_MIN + 15),
                random.uniform(MAP_MAX - 15, MAP_MAX - 8)
            ])
        else:
            # closer goals, easy to learn
            goal_distance = random.uniform(2.0, 12.0)
            goal_angle = random.uniform(-np.pi, np.pi)
            goal_x = np.clip(start_x + goal_distance * np.cos(goal_angle), MAP_MIN + 5, MAP_MAX - 5)
            goal_y = np.clip(start_y + goal_distance * np.sin(goal_angle), MAP_MIN + 5, MAP_MAX - 5)
        goal_theta = random.uniform(-np.pi, np.pi)

        start_pose = (start_x, start_y, start_theta)
        goal_pose = (goal_x, goal_y, goal_theta)

        # obstacle generation (centered boxes)
        boxes = generate_complex_obstacles(goal_pose, car_dims=CAR_DIMS, map_dims=(MAP_MIN, MAP_MAX))

        # generate trajectory
        traj = generate_reeds_shepp_trajectory(start_pose, goal_pose, TRAJ_LEN)

        # reject scenarios where the expert trajectory collides with obstacles
        if trajectory_collides(traj, start_pose, boxes):
            continue


        # we have collision free trajectory, 
        # trasform to local vehicle frame
        dx = (goal_x - start_x) * np.cos(-start_theta) - (goal_y - start_y) * np.sin(-start_theta)
        dy = (goal_x - start_x) * np.sin(-start_theta) + (goal_y - start_y) * np.cos(-start_theta)
        dtheta = goal_theta - start_theta

        # construct 6D relative target state: (dx, dy, cos_th, sin_th, target_v=0, target_delta=0)
        rel_pose_6d = np.array([dx, dy, np.cos(dtheta), np.sin(dtheta), 0.0, 0.0], dtype=np.float32)

        # create point cloud scan
        scan = ray_box_intersection((start_x, start_y), ray_angles + start_theta, boxes)

        # update samples
        trajectories.append(traj)
        point_cloud_scans.append(scan)
        rel_target_poses.append(rel_pose_6d)
        start_poses.append(start_pose)
        goal_poses.append(goal_pose)
        obstacle_lists.append(boxes)
    
        if len(trajectories) % 1000 == 0:
            print (f'-> {len(trajectories)} generated...')

    if len(trajectories) < num_samples:
        print(f"Warning: only generated {len(trajectories)}/{num_samples} collision-free scenarios.")
        
    # convert to tensors
    traj_tensor = torch.from_numpy(np.array(trajectories, dtype=np.float32))    
    point_cloud_tensor = torch.from_numpy(np.array(point_cloud_scans, dtype=np.float32)).unsqueeze(1) 
    pose_tensor = torch.from_numpy(np.array(rel_target_poses, dtype=np.float32))
    
    # add stats, for normalization
    stats = {
        'traj_mean': traj_tensor.mean(dim=(0, 1), keepdim=True),
        'traj_std': traj_tensor.std(dim=(0, 1), keepdim=True) + 1e-6,
        'point_cloud_mean': point_cloud_tensor.mean(),
        'point_cloud_std': point_cloud_tensor.std() + 1e-6,
        'pose_mean': pose_tensor.mean(dim=0, keepdim=True),
        'pose_std': pose_tensor.std(dim=0, keepdim=True) + 1e-6,
    }
    
    # format the data
    dataset = {
        'trajectories': traj_tensor,
        'point_cloud_scans': point_cloud_tensor,
        'rel_target_poses': pose_tensor,
        'start_poses': start_poses,
        'goal_poses': goal_poses,
        'boxes': obstacle_lists,
        'stats': stats
    }
    
    # export, in a monalitic tensor file
    torch.save(dataset, save_path)
    print(f"Dataset successfully generated and saved to {save_path}")

if __name__ == "__main__":
    # genrate the dataset
    create_parking_dataset(num_samples=NR_OF_SAMPLES_TO_GENERATE, save_path=DATASET_NAME)