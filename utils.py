##########################################
# Utils
# https://github.com/fvilmos
##########################################

import numpy as np
import cv2
from constants import *

class PointCloudSimulator:
    """
    Ray-maches 360 PointCloud beams against rectangular obstacles.
    Simulates a LiDAR behaviour.
    """
    def __init__(self, num_beams=360, max_range=20.0):
        self.num_beams = num_beams
        self.max_range = max_range
        self.angles = np.linspace(0, 2 * np.pi, num_beams, endpoint=False)

    def get_scan(self, ego_pose, obstacles):
        """
        Takes ego pose and obstacle list and generate the 2d point cloud from it
        """
        x, y, yaw = ego_pose
        scan = np.full(self.num_beams, 0.0)
        if not obstacles:
            return scan

        beam_angles = yaw + self.angles
        dx, dy = np.cos(beam_angles), np.sin(beam_angles)
        obs_arr = np.array(obstacles)
        ox, oy, ow, oh = obs_arr[:, 0], obs_arr[:, 1], obs_arr[:, 2], obs_arr[:, 3]

        dists = np.linspace(0.1, self.max_range, 50)
        px = x + dx[:, np.newaxis] * dists[np.newaxis, :]
        py = y + dy[:, np.newaxis] * dists[np.newaxis, :]

        hits = (np.abs(px[:, :, np.newaxis] - ox) <= ow / 2) & (np.abs(py[:, :, np.newaxis] - oy) <= oh / 2)
        any_hit = np.any(hits, axis=2)
        has_hit = np.any(any_hit, axis=1)
        scan[has_hit] = dists[np.argmax(any_hit, axis=1)[has_hit]]
        return scan


def world_to_pixel(x, y, map_min, map_max, img_size):
    """
    Convert world coordinates to pixel coordinates
    """
    map_range = map_max - map_min
    px = int((x - map_min) / map_range * img_size)
    py = int((img_size - (y - map_min) / map_range * img_size))
    return px, py


def draw_rect_rotated(img, cx, cy, width, height, angle, color, thickness=1):
    """
    Draw rotated rectangle on image using cv2 (negate angle for image coordinates)
    """
    angle_deg = -np.degrees(angle)
    rect = ((cx, cy), (width, height), angle_deg)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    cv2.polylines(img, [box], True, color, thickness)


def draw_filled_rect_rotated(img, cx, cy, width, height, angle, color, alpha=0.3):
    """
    Draw filled rotated rectangle with transparency
    """
    overlay = img.copy()
    angle_deg = -np.degrees(angle)  # Negate for OpenCV image coords (y-axis down)
    rect = ((cx, cy), (width, height), angle_deg)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    cv2.fillPoly(overlay, [box], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_arrow(img, x, y, angle, length=20, color=(0, 255, 255), thickness=2):
    """
    Draw heading arrow (negate angle for image coordinates with y-axis down)
    """
    angle_img = -angle  # Negate for OpenCV image coords
    end_x = int(x + length * np.cos(angle_img))
    end_y = int(y + length * np.sin(angle_img))
    cv2.arrowedLine(img, (int(x), int(y)), (end_x, end_y), color, thickness, tipLength=0.3)

def normalize_angle(angle):
    """
    Wrap angle to [-pi, pi]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def kinematic_bicycle(state, speed, steering, gear, wheelbase, dt):
    """
    Updates the car state using the simplified Kinematic Bicycle Model.
    v is negative in reverse gear, driving the vehicle backward.

    Patameters:
    -----------
    :state: location and orientation [x,y,theta]
    :speed: max speed alloved [m/s]
    :steering: steering angle [rad]
    :gear: direction 1,-1 - FW/Back [float]
    :wheelbase: distance between front rear axels [float, m]
    :dt: measurement time [float, time]
    """
    x, y, theta = state

    # positive forward, negative backward
    v = speed * gear

    # compute discrete next position, based on velocity and time
    dx = v * np.cos(theta) * dt
    dy = v * np.sin(theta) * dt

    # turning geometry
    d_theta = (v / wheelbase) * np.tan(steering) * dt

    # compute new location , orientation
    return [x + dx, y + dy, theta + d_theta]


def split_into_gear_segments(coarse_global, goal_pose=None, min_seg_len=3):
    """
    Splits waypoints into uniform-gear segments (FWD: +1.0, REV: -1.0)
    """
    if len(coarse_global) < 2:
        return [(1.0, np.array(coarse_global, dtype=np.float64))]

    v_raw = np.array([wp[3] if len(wp) > 3 else 0.0 for wp in coarse_global])
    v_smooth = np.convolve(v_raw, np.ones(3) / 3.0, mode='same')
    raw_gears = np.array([1.0 if v >= 0.0 else -1.0 for v in v_smooth])

    segments, curr_seg, curr_gear = [], [coarse_global[0]], raw_gears[0]
    
    for i in range(1, len(coarse_global)):
        if raw_gears[i] == curr_gear:
            curr_seg.append(coarse_global[i])
        else:
            if len(curr_seg) >= 2:
                segments.append((curr_gear, np.array(curr_seg, dtype=np.float64)))
            curr_gear = raw_gears[i]
            curr_seg = [coarse_global[i - 1], coarse_global[i]]

    if len(curr_seg) >= 2:
        segments.append((curr_gear, np.array(curr_seg, dtype=np.float64)))

    merged = []
    for g, wps in segments:
        if len(merged) > 0 and (merged[-1][0] == g or len(wps) < min_seg_len):
            prev_g, prev_wps = merged[-1]
            merged[-1] = (prev_g, np.vstack([prev_wps, wps[1:]]))
        else:
            merged.append((g, wps))

    if len(merged) > 1 and len(merged[-1][1]) < min_seg_len:
        prev_g, prev_wps = merged[-2]
        merged[-2] = (prev_g, np.vstack([prev_wps, merged[-1][1][1:]]))
        merged.pop()

    if goal_pose is not None and len(merged) > 0:
        merged[-1][1][-1, :3] = np.array(goal_pose[:3], dtype=np.float64)

    return merged


def align_terminal_slot_axis(coarse_wps, goal_pose, tail_dist=1.5):
    """
    Aligns final waypoints with stall centerline.
    """
    wps = np.array(coarse_wps, dtype=np.float64).copy()
    if goal_pose is None or len(wps) < 2:
        return wps

    gx, gy, gth = goal_pose[:3]
    dists = np.hypot(wps[:, 0] - gx, wps[:, 1] - gy)
    tail_mask = dists < tail_dist
    if np.sum(tail_mask) < 2:
        return wps

    ux, uy = np.cos(gth), np.sin(gth)
    tail_indices = np.where(tail_mask)[0]
    for idx in tail_indices:
        d = dists[idx]
        blend = (1.0 - (d / tail_dist)) ** 2
        dx = wps[idx, 0] - gx
        dy = wps[idx, 1] - gy
        s = dx * ux + dy * uy
        proj_x = gx + s * ux
        proj_y = gy + s * uy
        wps[idx, 0] = (1 - blend) * wps[idx, 0] + blend * proj_x
        wps[idx, 1] = (1 - blend) * wps[idx, 1] + blend * proj_y

        diff_fwd = abs(normalize_angle(wps[idx, 2] - gth))
        diff_rev = abs(normalize_angle(wps[idx, 2] - (gth + np.pi)))
        target_th = gth if diff_fwd < diff_rev else normalize_angle(gth + np.pi)
        th_diff = normalize_angle(target_th - wps[idx, 2])
        wps[idx, 2] = normalize_angle(wps[idx, 2] + blend * th_diff)

    return wps

def densify_segment(seg_wps, step_dist=0.01):
    """
    Densifies waypoints along a path segment with uniform arc-length parameterization.
    """
    wps = np.asarray(seg_wps, dtype=np.float64)
    if len(wps) < 2:
        return wps.copy()

    dists = np.hypot(np.diff(wps[:, 0]), np.diff(wps[:, 1]))
    valid = np.concatenate([[True], dists > 1e-4])
    wps = wps[valid]
    if len(wps) < 2:
        return wps.copy()

    dists = np.hypot(np.diff(wps[:, 0]), np.diff(wps[:, 1]))
    s_nodes = np.concatenate([[0.0], np.cumsum(dists)])
    total_len = s_nodes[-1]

    if total_len < 1e-3:
        return wps.copy()

    n_samples = max(4, int(np.ceil(total_len / step_dist)))
    s_eval = np.linspace(0.0, total_len, n_samples)

    new_x = np.interp(s_eval, s_nodes, wps[:, 0])
    new_y = np.interp(s_eval, s_nodes, wps[:, 1])
    new_yaw = normalize_angle(np.interp(s_eval, s_nodes, np.unwrap(wps[:, 2])))
    new_v = np.interp(s_eval, s_nodes, wps[:, 3]) if wps.shape[1] > 3 else np.zeros_like(s_eval)
    new_delta = np.interp(s_eval, s_nodes, wps[:, 4]) if wps.shape[1] > 4 else np.zeros_like(s_eval)

    return np.column_stack([new_x, new_y, new_yaw, new_v, new_delta])

def simulate_pure_pursuit_follow(coarse_waypoints,sim_dt=0.02, goal_override=None, k_e=1.8, k_d=0.15, k_i=0.05, v_soft=0.30, **kwargs):
    """
    Simulates closed-loop vehicle execution along coarse FlowNet waypoints
    """
    coarse = np.asarray(coarse_waypoints, dtype=np.float64)
    if len(coarse) < 2:
        return coarse.copy()

    wheelbase, max_steer = WHEELBASE, np.radians(35)
    goal_pose = np.array(goal_override, dtype=np.float64)[:3] if goal_override is not None else coarse[-1, :3].copy()

    coarse = align_terminal_slot_axis(coarse, goal_pose, tail_dist=1.5)
    segments = split_into_gear_segments(coarse, goal_pose, min_seg_len=3)

    trajectory = []
    state = np.array(coarse[0, :3], dtype=np.float64)

    for seg_i, (gear, seg_wps) in enumerate(segments):
        if len(seg_wps) < 2:
            continue

        dense_path = densify_segment(seg_wps, step_dist=0.01)

        if gear > 0:
            ref_path = dense_path.copy()
            ref_path[:, 0] += wheelbase * np.cos(dense_path[:, 2])
            ref_path[:, 1] += wheelbase * np.sin(dense_path[:, 2])
        else:
            ref_path = dense_path.copy()

        dists_s = np.hypot(np.diff(ref_path[:, 0]), np.diff(ref_path[:, 1]))
        s_arr = np.concatenate([[0.0], np.cumsum(dists_s)])
        yaw_unwrapped = np.unwrap(ref_path[:, 2])
        dyaw_ds = np.gradient(yaw_unwrapped, s_arr, edge_order=2) if len(s_arr) > 2 else np.zeros_like(s_arr)
        curv_steer = np.arctan(wheelbase * dyaw_ds)

        curr_idx = 0
        seg_len = s_arr[-1]
        max_dist = max(0.5, seg_len * 1.5 + 0.5)
        dist_driven = 0.0
        max_steps = max(len(ref_path) * 8, 2000)
        prev_e_ct = 0.0
        sum_e_ct = 0.0

        for step in range(max_steps):
            if gear > 0:
                axle_x = state[0] + wheelbase * np.cos(state[2])
                axle_y = state[1] + wheelbase * np.sin(state[2])
            else:
                axle_x = state[0]
                axle_y = state[1]

            search_start = max(0, curr_idx - 5)
            search_end = min(curr_idx + 80, len(ref_path))
            search_window = ref_path[search_start:search_end]
            dists = np.hypot(search_window[:, 0] - axle_x, search_window[:, 1] - axle_y)
            curr_idx = search_start + int(np.argmin(dists))

            dx_end = dense_path[-1, 0] - state[0]
            dy_end = dense_path[-1, 1] - state[1]
            dist_to_seg_end = np.hypot(dx_end, dy_end)
            lx_end = dx_end * np.cos(state[2]) + dy_end * np.sin(state[2])

            passed_plane = (gear * lx_end) <= 0.0
            near_end = curr_idx >= len(ref_path) - 50 or dist_to_seg_end < 0.80
            if (near_end and passed_plane) or (curr_idx >= len(ref_path) - 5 and dist_to_seg_end < 0.15) or dist_driven >= max_dist:
                trajectory.append([state[0], state[1], state[2], 0.0, 0.0])
                break

            path_pt = ref_path[curr_idx]
            path_yaw = path_pt[2]
            dx_p = axle_x - path_pt[0]
            dy_p = axle_y - path_pt[1]
            e_ct = -dx_p * np.sin(path_yaw) + dy_p * np.cos(path_yaw)

            e_ct_dot = (e_ct - prev_e_ct) / sim_dt if step > 0 else 0.0
            prev_e_ct = e_ct
            sum_e_ct = np.clip(sum_e_ct + e_ct * sim_dt, -0.20, 0.20)

            delta_ff = curv_steer[curr_idx] if curr_idx < len(curv_steer) else 0.0
            ct_term = k_e * e_ct + k_d * e_ct_dot + k_i * sum_e_ct

            if gear > 0:
                th_e = normalize_angle(path_yaw - state[2])
                steer = np.clip(0.50 * delta_ff + th_e + np.arctan2(-ct_term, v_soft), -max_steer, max_steer)
            else:
                rev_path_yaw = normalize_angle(path_yaw + np.pi)
                th_e = normalize_angle(rev_path_yaw - state[2])
                steer = np.clip(0.50 * delta_ff + th_e + np.arctan2(ct_term, v_soft), -max_steer, max_steer)

            speed = min(0.6, max(0.06, dist_to_seg_end * 1.5))
            dist_driven += speed * sim_dt
            trajectory.append([state[0], state[1], state[2], speed * gear, steer])
            state = kinematic_bicycle(state, speed, steer, gear, wheelbase, sim_dt)
            state[2] = normalize_angle(state[2])

    for _ in range(15):
        trajectory.append([state[0], state[1], state[2], 0.0, 0.0])

    return np.array(trajectory, dtype=np.float32)

def generate_complex_obstacles(goal, car_dims=[2.0,4.5], map_dims=[-25,25]):
    """
    Generated cars, small objects and walls
    
    Parameters:
    ----------
    goal - location to reach, [x,y,theta]
    car+dims - width, lenght in meter
    map_dims - min max of the map, [xmin=y_min,x_max=y_max]

    Returns:
    -------
    list of elements of [cx,cy,w,h]
    """
    obstacles = []

    # generate obstacles, "cars"
    for rx in [-10, 10]:
        for ry in range(-20, 21, 5):
            # protect goal
            if np.sqrt((rx - goal[0])**2 + (ry - goal[1])**2) < 3.0:
                continue
            # protect center
            if np.abs(rx) <= car_dims[0]+0.2 or np.abs(ry) <= car_dims[1]+0.2:
                continue

            if np.random.random() > 0.3:
                obstacles.append((rx, ry, np.random.choice([1.2, 3.5, 4.5]), np.random.choice([2.0, 2.5, 1.1])))
    
    # generate walls, small objects
    for _ in range(np.random.randint(2, 6)):
        type = np.random.random()
        ox, oy = np.random.uniform(map_dims[0], map_dims[1]), np.random.uniform(map_dims[0], map_dims[1])
        
        # protect goal
        if np.sqrt((ox - goal[0])**2 + (oy - goal[1])**2) < 3.0: 
            continue
        
        # protect center
        if np.abs(ox) <= car_dims[0]+0.2 or np.abs(oy) <= car_dims[1]+0.2:
            continue
        if type < 0.3: obstacles.append((ox, oy, 0.3, 0.3))
        elif type < 0.6: 
            w, h = (10.0, 0.5) if np.random.random() > 0.5 else (0.5, 10.0)
            obstacles.append((ox, oy, w, h))
        else: obstacles.append((ox, oy, np.random.uniform(1, 4), np.random.uniform(1, 4)))
    # list of: [[cx,cy,w,h]]
    return obstacles


def footprint_collides(state, obstacles, margin=0.1):
    """
    Check if a vehicle state collides with any obstacle
    
    Parameters:
    ----------
    state: [x, y, theta] in global frame
    obstacles: list of (cx, cy, w, h)
    """
    x, y, theta = state
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    hl = CAR_DIMS[0] / 2.0 + margin
    hw = CAR_DIMS[1] / 2.0 + margin

    # footprint corners in global frame
    corners_local = np.array([
        [hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]
    ])
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    corners = (R @ corners_local.T).T + np.array([x, y])

    # check for all obstacles
    for cx, cy, w, h in obstacles:
        x1, x2 = cx - w / 2.0, cx + w / 2.0
        y1, y2 = cy - h / 2.0, cy + h / 2.0
        if np.any((corners[:, 0] >= x1) & (corners[:, 0] <= x2) &
                  (corners[:, 1] >= y1) & (corners[:, 1] <= y2)):
            return True
    return False


def trajectory_collides(traj_local, start_pose, obstacles):
    """
    Check a local-frame trajectory against global obstacles

    Parameters:
    ----------
    traj_local: [T, 6] with [lx, ly, cos_th, sin_th, v, delta]
    start_pose:: [x, y, theta] in global frame
    obstacles: list of (cx, cy, w, h)
    """
    x0, y0, th0 = start_pose
    cos0, sin0 = np.cos(th0), np.sin(th0)
    for point in traj_local:
        lx, ly = point[0], point[1]
        gx = x0 + lx * cos0 - ly * sin0
        gy = y0 + lx * sin0 + ly * cos0
        gth = th0 + np.arctan2(point[3], point[2])
        if footprint_collides((gx, gy, gth), obstacles):
            return True
    return False

def ray_box_intersection(ray_origin, ray_angles, boxes, max_range=MAX_POINT_CLOUD_RANGE):
    """
    Computes 360-degree point cloud distances against centered-box obstacles
    
    Parameters:
    ----------
    ray_origin: (x, y)
    ray_angles: array of 360 angles in radians
    boxes: list of (cx, cy, w, h) obstacles
    """
    ox, oy = ray_origin
    ranges = np.full(len(ray_angles), max_range, dtype=np.float32)

    # add lot boundaries
    wall_thickness = 1.0
    all_boxes = list(boxes) + [
        (MAP_MIN - wall_thickness / 2.0, 0.0, wall_thickness, LOT_SIZE + 2.0 * wall_thickness),
        (MAP_MAX + wall_thickness / 2.0, 0.0, wall_thickness, LOT_SIZE + 2.0 * wall_thickness),
        (0.0, MAP_MIN - wall_thickness / 2.0, LOT_SIZE + 2.0 * wall_thickness, wall_thickness),
        (0.0, MAP_MAX + wall_thickness / 2.0, LOT_SIZE + 2.0 * wall_thickness, wall_thickness),
    ]

    for i, alpha in enumerate(ray_angles):
        dx = np.cos(alpha)
        dy = np.sin(alpha)
        min_t = max_range

        for (cx, cy, w, h) in all_boxes:
            x1, x2 = cx - w / 2.0, cx + w / 2.0
            y1, y2 = cy - h / 2.0, cy + h / 2.0

            tx1 = (x1 - ox) / (dx + 1e-8)
            tx2 = (x2 - ox) / (dx + 1e-8)
            tmin_x, tmax_x = min(tx1, tx2), max(tx1, tx2)

            ty1 = (y1 - oy) / (dy + 1e-8)
            ty2 = (y2 - oy) / (dy + 1e-8)
            tmin_y, tmax_y = min(ty1, ty2), max(ty1, ty2)

            t_near = max(tmin_x, tmin_y)
            t_far = min(tmax_x, tmax_y)

            if t_near <= t_far and t_far > 0:
                hit_dist = t_near if t_near > 0 else 0.0
                if hit_dist < min_t:
                    min_t = hit_dist

        ranges[i] = min_t

    return ranges

def extract_steering(curvature):
    """
    Approximate steering angle from path curvature for a kinematic bicycle
        
    Parameters:
    ----------
    curvature: signed path curvature from rsplan
    """
    # radius = 1 / |curvature| 
    # delta = atan2(wheelbase * curvature, 1)
    if abs(curvature) < 1e-6:
        return 0.0
    return float(np.arctan(WHEELBASE * curvature))


def compute_smooth_velocity_profile(wps, max_v=MAX_VELOCITY, a_max=0.8, a_lat_max=0.5):
    """
    Computes a continuous trapezoidal velocity profile along Reeds-Shepp waypoints:
    - 0 speed at start, cusps (direction reversals), and final goal.
    - Curvature-limited lateral acceleration (v <= sqrt(a_lat_max / kappa)).
    - Forward and backward longitudinal acceleration integration.
    """
    N = len(wps)
    if N <= 1:
        return np.zeros(N, dtype=np.float32)

    directions = wps[:, 3]
    curvatures = wps[:, 4]

    dx = np.diff(wps[:, 0])
    dy = np.diff(wps[:, 1])
    ds = np.hypot(dx, dy)

    v_curve = np.full(N, max_v, dtype=np.float32)
    for i in range(N):
        k = abs(curvatures[i])
        if k > 1e-4:
            v_curve[i] = min(max_v, float(np.sqrt(a_lat_max / k)))

    # Forward pass: start and cusps have 0 speed
    v_fwd = np.zeros(N, dtype=np.float32)
    for i in range(1, N):
        if directions[i] != directions[i - 1]:
            v_fwd[i] = 0.0
        else:
            v_fwd[i] = min(v_curve[i], float(np.sqrt(v_fwd[i - 1]**2 + 2.0 * a_max * ds[i - 1])))

    # Backward pass: goal and cusps have 0 speed
    v_bwd = np.zeros(N, dtype=np.float32)
    for i in range(N - 2, -1, -1):
        if directions[i] != directions[i + 1]:
            v_bwd[i] = 0.0
        else:
            v_bwd[i] = min(v_curve[i], float(np.sqrt(v_bwd[i + 1]**2 + 2.0 * a_max * ds[i])))

    speed = np.minimum(v_fwd, v_bwd)
    v_signed = speed * directions
    return v_signed.astype(np.float32)

