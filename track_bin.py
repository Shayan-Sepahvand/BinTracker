"""
Skyscouter – Computer Vision Engineer Technical Assessment
Skeleton file: track_bin.py  (v2 – updated with geometry guidance)
Restructure freely. This is a starting point only.
"""

import cv2
import json
import argparse
import time
import numpy as np
import os
import csv
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ── Known target dimensions ──────────────────────────────────────────────────
BIN_DIAMETER_M = 0.40   # standard outdoor garbage bin
BIN_HEIGHT_M   = 0.65

# ── Calibration loader ────────────────────────────────────────────────────────

def load_calib(path: str):
    """
    Load camera intrinsics and mount geometry from calib.json.
    Returns:
        K         (3×3 ndarray)  intrinsic matrix
        D         (5,  ndarray)  distortion coefficients [k1,k2,p1,p2,k3]
        cam_h     (float)        camera height above ground in metres
        tilt_rad  (float)        downward tilt in radians (negative = down)
    """
    with open(path) as f:
        c = json.load(f)
    K        = np.array(c["K"],           dtype=np.float64)
    D        = np.array(c["dist_coeffs"], dtype=np.float64)
    cam_h    = float(c["camera_height_m"])
    tilt_rad = float(np.deg2rad(c["camera_tilt_deg"]))
    return K, D, cam_h, tilt_rad

# ── Waypoint loader ────────────────────────────────────────────────────────
def load_waypoints(path: str):
    """
    Load 2D ground-truth pixel coordinates and their corresponding frame index from waypoint.json
    Returns:
        An np array containing (3 by 4)  [pixel_u, pixel_v, order, approx_frame]
    """
    with open(path, "r") as f:
        data = json.load(f)
    markers = data["markers"]
    waypoints = []
    for i, marker in enumerate(markers):
        u = int(marker["pixel_u"])
        v = int(marker["pixel_v"])
        frame_idx = int(marker["approx_frame"])
        waypoints.append([u, v, i, frame_idx]) 
    return np.array(waypoints, dtype=np.float64)

# ── Coordinate transforms ─────────────────────────────────────────────────────

def build_extrinsic(cam_h: float, tilt_rad: float):

    """
    Construct the extrinsics of the camera using the give fixed static tf.
    Returns:
        The rotation matrix that transforms from camera to the spatial (world frame).
        The rotation from the tilted camera {frame 2} to untilted camera {frame 1} is Rx,
        The rotation from the untilted camera {frame 1} to the world frame {frame 0} is static here.
        The rotation from the the tilted camera frame {frame 2} to the world frame {frame 0} is Rs @ Rx.
        The world postion vector should be described w.r.t world frame, thus, Pworld = Rs @ Rx @ Pcam + t 
    """
    # --- Axis mapping: camera -> world ---
    Rs = np.array([
        [ 0,  0,  1],   # world_X = cam_Z
        [-1,  0,  0],   # world_Y = -cam_X
        [ 0, -1,  0],   # world_Z = -cam_Y
    ], dtype=np.float64)

    # --- Rotation in CAMERA frame (pitch around X) ---
    c, s = np.cos(tilt_rad), np.sin(tilt_rad)
    Rx = np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c],
    ], dtype=np.float64)

    R = Rs @ Rx
    t = np.array([0.0, 0.0, cam_h]) #this is the postion vec. that starts from the world and ends at the camera optical centre

    return R, t

def cam_to_world(xyz_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    The world postion vector should be described w.r.t world frame, thus, Pworld = Rs @ Rx @ Pcam + t 

    """
    return R @ xyz_cam + t


# ── Detection ────────────────────────────────────────────────────────────────

def load_detector(use_gpu: bool = True, wights_path: str = None):
    """
    Load and return your detector.
    If use_gpu=True, configure the model to use GPU.
    Document GPU model, VRAM, and CUDA version in README if use_gpu=True.
    """
    model = YOLO(wights_path) 
    if use_gpu:
        model.to("cuda")
    else:
        model.to("cpu")
    return model


def detect_bin(frame: np.ndarray, model) -> tuple | None:
    """
    Detect the garbage bin in a single BGR frame.
    Returns (x1, y1, x2, y2, confidence) or None if not detected.
    """
    results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
    
    best_det = None
    max_conf = -1.0

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            conf = float(box.conf[0])
            
            if conf > max_conf:
                max_conf = conf
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                best_det = (x1, y1, x2, y2, conf)

    return best_det


def plot_inference_performance(times, args):

    """
    plots the inference time in ms vs the frame #
    plot saved to {args.output}/inference_time_ms.png
    """
    plt.figure(figsize=(10, 5))
    plt.plot(times[1:], label='Inference Time (ms)', color='blue', linewidth=1)
    avg_time = sum(times[1:]) / len(times)
    plt.axhline(y=avg_time, color='red', linestyle='--', label=f'Avg: {avg_time:.2f}ms')
    plt.title('YOLOv10 Inference Performance')
    plt.xlabel('Frame Number')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output, "inference_perf.png"))
    # print(f"[run.sh] Performance plot saved to {args.output}/inference_time_ms.png")


def draw_bounding_box(frame, x1, y1, x2, y2, conf):
    """Draws a green bounding box and confidence score on the frame."""
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    label = f"Bin: {conf:.2f}"
    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
    cv2.putText(frame, label, (int(x1), int(label_y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# ── 3D localisation ───────────────────────────────────────────────────────────

def estimate_3d(bbox, K, D, bin_height_m=0.65):
    # Extract BBox data
    u_dist = (bbox[0] + bbox[2]) / 2.0
    v_dist = (bbox[1] + bbox[3]) / 2.0
    h_px = np.abs(bbox[3] - bbox[1])

    # Extract Intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = D

    # 1. Depth (Z) calculation
    # Derived from: Z = (fy * Real_Height) / Pixel_Height
    Z = (fy * bin_height_m) / h_px

    # 2. Convert pixel to normalized distorted coordinates
    x_d = (u_dist - cx) / fx
    y_d = (v_dist - cy) / fy

    # 3. Iterative Undistortion (Reversing the D coefficients)
    # We start by assuming the undistorted point (x, y) is the distorted one
    x, y = x_d, y_d
    
    for _ in range(5): # 5 iterations is standard for high precision
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r2*r4
        
        # Radial distortion term
        radial = 1 + k1*r2 + k2*r4 + k3*r6
        
        # Tangential distortion terms
        dx = 2*p1*x*y + p2*(r2 + 2*x*x)
        dy = p1*(r2 + 2*y*y) + 2*p2*x*y
        
        # Update estimate: x_undist = (x_distorted - tangential) / radial
        x = (x_d - dx) / radial
        y = (y_d - dy) / radial

    # 4. Project back to Camera 3D Space
    X = x * Z
    Y = y * Z
    
    return np.array([X, Y, Z])

# ── 3D localisation (bonus) ───────────────────────────────────────────────────────────

def estimate_3d_wp(K, D, waypoint, Z_frame):

    u_dist = waypoint[0]
    v_dist = waypoint[1]

    # Extract Intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = D


    # 2. Convert pixel to normalized distorted coordinates
    x_d = (u_dist - cx) / fx
    y_d = (v_dist - cy) / fy

    # 3. Iterative Undistortion (Reversing the D coefficients)
    # We start by assuming the undistorted point (x, y) is the distorted one
    x, y = x_d, y_d
    
    for _ in range(5): # 5 iterations is standard for high precision
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r2*r4
        
        # Radial distortion term
        radial = 1 + k1*r2 + k2*r4 + k3*r6
        
        # Tangential distortion terms
        dx = 2*p1*x*y + p2*(r2 + 2*x*x)
        dy = p1*(r2 + 2*y*y) + 2*p2*x*y
        
        # Update estimate: x_undist = (x_distorted - tangential) / radial
        x = (x_d - dx) / radial
        y = (y_d - dy) / radial

    # 4. Project back to Camera 3D Space
    X = x * Z_frame
    Y = y * Z_frame
    
    return np.array([X, Y, Z_frame])


def plot_world_trajectory(world_traj, stop_positions, save_path="trajectory.png"):
    """
    Plots top-down XY trajectory of the bin in world frame.

    Parameters:
    -----------
    world_traj : array-like (N, 3)
        Full trajectory points in world frame [x, y, z].

    stop_positions : array-like (M, 3)
        Stop positions in world frame [x, y, z].

    save_path : str
        Output image path.
    """

    world_traj = np.asarray(world_traj)
    stop_positions = np.asarray(stop_positions)

    if world_traj.shape[0] == 0:
        raise ValueError("world_traj is empty — cannot plot trajectory.")

    plt.figure()

    # --- trajectory ---
    plt.plot(world_traj[:, 0], world_traj[:, 1], '-b', label="Trajectory")

    # --- start / end ---
    plt.scatter(world_traj[0, 0], world_traj[0, 1],
                c='green', s=80, label="Start")

    plt.scatter(world_traj[-1, 0], world_traj[-1, 1],
                c='red', s=80, label="End")

    # --- stop positions ---
    if stop_positions.size > 0:
        plt.scatter(stop_positions[:, 0], stop_positions[:, 1],
                    c='orange', s=120, marker='x', label="Stops")

    plt.xlabel("World X (m)")
    plt.ylabel("World Y (m)")
    plt.title("Bin Trajectory (Top-Down XY View)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



def rmse_per_point(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute RMSE (Euclidean error) for each corresponding 3D point.

    Args:
        A: (N, 3) array of ground truth points
        B: (N, 3) array of estimated points

    Returns:
        errors: (N,) array where each entry is RMSE for that point
    """
    if A.shape != B.shape:
        raise ValueError("Input arrays must have the same shape")
    if A.shape[1] != 3:
        raise ValueError("Each point must be 3D")

    diff = A - B                      # (N, 3)
    errors = np.sqrt(np.mean(diff**2, axis=1))  # RMSE per point

    return errors

# ── Optional: Kalman filter ───────────────────────────────────────────────────

class PositionKalman:
    """
    Constant-velocity Kalman filter over 3D world position.
    State vector: [x, y, z, vx, vy, vz]

    Implement this for the bonus task (3c).
    Explain Q, R, and state vector choices in README

    Constant-velocity Kalman filter over 3D world position.

    State:
        x = [x, y, z, vx, vy, vz]^T

    Measurement:
        z = [x, y, z]^T
    """

    def __init__(self, dt=1/30): # Set to 1/30 for standard 30 FPS video
            self.dt = dt

            # --- State vector (6x1) ---
            self.x = np.zeros((6, 1), dtype=np.float64)

            # Initial uncertainty: 
            # Make it VERY uncertain at the start so it snaps to the first measurement instantly
            self.P = np.diag([10, 10, 10, 100, 100, 100])

            # --- State transition model (constant velocity) ---
            self.F = np.array([
                [1, 0, 0, dt, 0,  0],
                [0, 1, 0, 0,  dt, 0],
                [0, 0, 1, 0,  0,  dt],
                [0, 0, 0, 1,  0,  0],
                [0, 0, 0, 0,  1,  0],
                [0, 0, 0, 0,  0,  1],
            ], dtype=np.float64)

            # --- Measurement model (we observe position only) ---
            self.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ], dtype=np.float64)

            # --- Process noise (Q) --- 
            # Keep q_pos tiny.
            # Drop q_vel to 0.1. This reduces the "momentum" so the filter 
            # stops immediately when the raw data stops, fixing the negative Y reduction.
            q_pos = 0.001 
            q_vel = 0.1  
            self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])

            # --- Measurement noise (R) ---
            # X: 0.1 (Standard smoothing)
            # Y: 0.01 (Very low! Tells the filter to trust the raw Y data because it is clean)
            # Z: 0.5 (High! Tells the filter to aggressively smooth the noisy depth estimations)
            self.R = np.diag([0.1, 0.01, 0.5])

    # -------------------------------------------------
    def predict(self) -> np.ndarray:
        """
        Predict next state.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x[:3].flatten()

    # -------------------------------------------------
    def update(self, xyz_meas: np.ndarray) -> np.ndarray:
        """
        Correct with measurement [x, y, z].
        """
        z = np.asarray(xyz_meas, dtype=np.float64).reshape(3, 1)

        # Innovation
        y = z - (self.H @ self.x)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

        return self.x[:3].flatten()

    # ── Plot raw vs filtered position ───────────────────────────────────────────────────

    def plot_raw_vs_filtered_position(self, raw_traj, filtered_traj, save_path="./results/trajectory_kf_vs_raw.png"):

        raw_traj = np.asarray(raw_traj)
        filtered_traj = np.asarray(filtered_traj)


        t = np.arange(raw_traj.shape[0])

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = ['X', 'Y', 'Z']

        for i in range(3):
            axs[i].plot(t, raw_traj[:, i], '--', label='Raw')
            axs[i].plot(t, filtered_traj[:, i], '-', label='Filtered')
            axs[i].set_ylabel(f"{labels[i]} (m)")
            axs[i].grid(True)
            axs[i].legend()

        axs[2].set_xlabel("Frame")
        fig.suptitle("Raw vs Kalman Filtered Trajectory")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def quantify_jitter_reduction(self, raw_trajectory, filtered_trajectory, start_frame, end_frame):
        """
        Calculates the standard deviation of stationary readings to quantify jitter.
        
        Args:
            raw_trajectory: List of raw [x, y, z] measurements.
            filtered_trajectory: List of Kalman filtered [x, y, z] estimates.
            start_frame: The frame index where the stationary period begins.
            end_frame: The frame index where the stationary period ends.
        """
        # 1. Convert to NumPy arrays
        raw_arr = np.array(raw_trajectory)
        filt_arr = np.array(filtered_trajectory)

        # 2. Extract the stationary window
        raw_stat = raw_arr[start_frame:end_frame]
        filt_stat = filt_arr[start_frame:end_frame]

        # 3. Calculate standard deviation for each axis (column-wise)
        raw_std = np.std(raw_stat, axis=0)
        filt_std = np.std(filt_stat, axis=0)

        # 4. Calculate the percentage reduction in jitter
        # Protect against division by zero just in case the raw data is perfectly flat
        reduction = np.zeros(3)
        for i in range(3):
            if raw_std[i] > 0:
                reduction[i] = ((raw_std[i] - filt_std[i]) / raw_std[i]) * 100.0

        # 5. Print out the formatted report
        axes = ['X (Forward)', 'Y (Left)   ', 'Z (Up)     ']
        print("=" * 45)
        print(f"JITTER REDUCTION (Frames {start_frame} to {end_frame})")
        print("=" * 45)
        print(f"{'Axis':<15} | {'Raw Std':<10} | {'Filt Std':<10} | {'Reduction'}")
        print("-" * 45)
        
        for i, axis in enumerate(axes):
            print(f"{axis:<15} | {raw_std[i]:<10.4f} | {filt_std[i]:<10.4f} | {reduction[i]:.1f}%")
        print("=" * 45)

        return raw_std, filt_std, reduction

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--calib', required=True)
    parser.add_argument('--output', default='results')
    parser.add_argument('--weights', default='./weights/best.pt') # Add this line!
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--kalman', action='store_true')
    args = parser.parse_args()

    model = load_detector(args.gpu, args.weights)
    K, D, cam_h, tilt_rad = load_calib(args.calib)
    R, t   = build_extrinsic(cam_h, tilt_rad)
    wps = load_waypoints("./waypoints.json")

    kf = PositionKalman(dt=1/30) if args.kalman else None

    

    trajectory = []
    trajectory_KF = []
    xyz_cam_traj = []
    stop_positions = []
    GT = []
    EST = []
    inference_times = []
    last_age   = 0
    no_missed_fr = 0

    cap = cv2.VideoCapture(args.video)
    save_path = os.path.join(args.output)
    save_path_detected_imgs = os.path.join(args.output, "./detections")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_detected_imgs, exist_ok=True)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    # Create the full path: e.g., "results/output.csv"
    csv_px_path = os.path.join(args.output, "1a.csv")
    csv_path = os.path.join(args.output, "2b.csv")
    csv_world_path = os.path.join(args.output, "2c.csv")
    # 1. Open CSV using the csv module for cleaner formatting
    with open(csv_path, "w", newline='') as csv_file, open(csv_world_path, "w", newline='') as csv_world, open(csv_px_path, "w", newline='') as csv_px:
        
        writer_px = csv.writer(csv_px)
        writer = csv.writer(csv_file)
        writer_world = csv.writer(csv_world)
        
        # Matches your requested header: frame_id, timestamp_ms, x_cam, y_cam, z_cam, confidence
        # Note: I removed world coords to match your specific format request, 
        # but you can add them back if needed.
        writer_px.writerow(["frame_id", "timestamp_ms", "u1", "v1", "u2", "v2"])
        writer.writerow(["frame_id", "timestamp_ms", "x_cam", "y_cam", "z_cam", "confidence"])
        writer_world.writerow(["frame_id", "t_ms", "x_cam", "y_cam", "z_cam", "x_world", "y_world", "z_world", "conf"])

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get timestamp directly from the video capture (in milliseconds)
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if kf is not None:
                predicted_pos = kf.predict()
            start_inf = time.perf_counter()
            det = detect_bin(frame, model) # Single detection call
            end_inf = time.perf_counter()
            dt_ms = int((end_inf - start_inf) * 1000)
            inference_times.append((dt_ms))

            if det is not None:
                last_age = 0 #reset the number of cons. missed frames
                x1, y1, x2, y2, conf = det

                frame = draw_bounding_box(frame, x1, y1, x2, y2, conf)

                filename = os.path.join(save_path_detected_imgs, f"frame_{frame_id:05d}.png")
                cv2.imwrite(filename, frame)                # 2. Calculate 3D coordinates
                xyz_cam = estimate_3d((x1, y1, x2, y2), K, D, BIN_HEIGHT_M)
                x_c, y_c, z_c = xyz_cam
                Pw = cam_to_world(xyz_cam, R, t)

                # 3. Write formatted data to CSV
                # frame_id:04d ensures the '0042' padding format

                writer_px.writerow([
                    f"{frame_id:04d}",
                    f"{timestamp_ms:.0f}", 
                    f"{x1}", 
                    f"{y1}", 
                    f"{x2}", 
                    f"{y2}"
                ])

                writer.writerow([
                    f"{frame_id:04d}",
                    f"{timestamp_ms:.0f}", 
                    f"{x_c:.2f}", 
                    f"{y_c:.2f}", 
                    f"{z_c:.2f}", 
                    f"{conf:.2f}"
                ])

                # --- Write WORLD CSV ---
                writer_world.writerow([
                    f"{frame_id:04d}",
                    f"{timestamp_ms:.0f}",
                    f"{x_c:.2f}",
                    f"{y_c:.2f}",
                    f"{z_c:.2f}",
                    f"{Pw[0]:.2f}",
                    f"{Pw[1]:.2f}",
                    f"{Pw[2]:.2f}",
                    f"{conf:.2f}"
                ])

                # --- STDOUT (REQUIRED FORMAT) ---
                print(
                    f"[{frame_id:03d}] bin @ world "
                    f"({Pw[0]:.2f}, {Pw[1]:.2f}, {Pw[2]:.2f}) m "
                    f"conf={conf:.2f} dt={dt_ms}ms",
                    flush=True
                )
                if kf is not None:
                    
                    filtered = kf.update(Pw)
                    trajectory_KF.append(filtered)

                trajectory.append(Pw)
                xyz_cam_traj.append(xyz_cam)

            else:
                no_missed_fr += 1

                last_age +=1
                print(
                    f"[{frame_id:03d}] OCCLUDED - last known"
                    f"({trajectory[-1][0]:.2f}, {trajectory[-1][1]:.2f}, {trajectory[-1][2]:.2f}) m "
                    f" age={last_age}fr",
                    flush=True
                )


            # 2d ============================================
            for i in range(len(wps)):
                u = wps[i, 0]
                v = wps[i, 1]
                frame_wp = int(wps[i, 3])  # <-- correct column now

                if frame_id == frame_wp:
                    xyz_cam = estimate_3d_wp(K, D, (u, v), xyz_cam_traj[-1][2])
                    Pw_est = cam_to_world(xyz_cam, R, t)
                    GT.append(Pw_est)
                    EST.append(Pw)
                    stop_positions.append(Pw_est)
            # end of 2d ============================================


            frame_id += 1

        cap.release()



    # 5. Print out the formatted report
    print("=" * 45)
    detection_rate = 100 * (1 - no_missed_fr / frame_id)
    print(f"[run.sh] Detection Rate is {detection_rate:.2f}%.")
    print("=" * 45)

    RMSE = rmse_per_point(np.array(GT), np.array(EST))
    print("=" * 45)
    print(f"[run.sh] RMSE per axis: x={RMSE[0]:.2f}, y={RMSE[1]:.2f}, z={RMSE[2]:.2f}")
    print("=" * 45)

    if inference_times:
        plot_inference_performance(inference_times, args)

    # end of 2d ============================================
    plot_world_trajectory(trajectory, stop_positions, save_path="./results/trajectory.png")
    if kf is not None:
        kf.plot_raw_vs_filtered_position(trajectory, trajectory_KF)
        kf.quantify_jitter_reduction(trajectory, trajectory_KF, 250, 280)
    # end of 2d ============================================



if __name__ == "__main__":
    main()





