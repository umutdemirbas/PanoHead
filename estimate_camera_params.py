"""
Estimate camera parameters (pose and intrinsics) for face images.
Uses MediaPipe for face detection and landmark estimation, then solves for camera matrix using PnP.
"""

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Optional: MediaPipe for better accuracy
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------

def estimate_camera_params(image_path, focal_length=600, verbose=False):
    """
    Estimate camera parameters from a single face image.
    Uses OpenCV's DNN face detector (no compilation needed).
    
    Args:
        image_path: Path to the image
        focal_length: Approximate focal length (in pixels)
        verbose: Print debug info
        
    Returns:
        camera_matrix: 4x4 camera-to-world matrix
        intrinsics: Camera intrinsics [fx, fy, cx, cy]
    """
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None, None
    
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Try MediaPipe first if available
    if HAS_MEDIAPIPE:
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import mediapipe as mpi
            
            base_options = python.BaseOptions(model_asset_path=None)
            options = vision.FaceMeshOptions(base_options=base_options)
            detector = vision.FaceMesh(options=options)
            
            mp_image = mpi.Image(image_format=mpi.ImageFormat.SRGB, data=image_rgb)
            detection_result = detector.detect(mp_image)
            
            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                landmark_indices = list(range(min(6, len(landmarks))))
                
                image_points_2d = []
                for idx in landmark_indices:
                    lm = landmarks[idx]
                    x = lm.x * width
                    y = lm.y * height
                    image_points_2d.append([x, y])
                
                image_points_2d = np.array(image_points_2d, dtype=np.float32)
                return _solve_pnp(image_points_2d, width, height, focal_length, image_path, verbose)
        except Exception as e:
            if verbose:
                print(f"MediaPipe failed: {e}, using Haar cascade")
    
    # Use Haar cascade for face detection (built-in, no downloads needed)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return None, None
    
    (x1, y1, w, h) = faces[0]
    x2, y2 = x1 + w, y1 + h
    
    # Extract face center and approximate landmarks
    face_center_x = (x1 + x2) / 2.0
    face_center_y = (y1 + y2) / 2.0
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Approximate 2D landmarks based on face bounding box
    image_points_2d = np.array([
        [face_center_x, face_center_y - face_height * 0.2],  # nose
        [face_center_x - face_width * 0.25, face_center_y - face_height * 0.1],  # left eye
        [face_center_x + face_width * 0.25, face_center_y - face_height * 0.1],  # right eye
        [face_center_x, face_center_y + face_height * 0.2],  # mouth
        [x1 + face_width * 0.1, face_center_y],  # left jaw
        [x2 - face_width * 0.1, face_center_y],  # right jaw
    ], dtype=np.float32)
    
    return _solve_pnp(image_points_2d, width, height, focal_length, image_path, verbose)


def _solve_pnp(image_points_2d, width, height, focal_length, image_path, verbose=False):
    """Helper to solve PnP given 2D points"""
    
    # 3D model points
    object_points_3d = np.array([
        [0.0, 0.0, 0.0],           # nose
        [-18.76, 12.40, 4.04],     # left eye
        [18.76, 12.40, 4.04],      # right eye
        [-9.0, -14.0, -2.0],       # mouth
        [-35.0, -25.0, 0.0],       # left jaw
        [35.0, -25.0, 0.0],        # right jaw
    ], dtype=np.float32)
    
    # Camera intrinsics
    camera_matrix = np.array([
        [focal_length, 0, width/2],
        [0, focal_length, height/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1))
    
    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        object_points_3d,
        image_points_2d,
        camera_matrix,
        dist_coeffs,
        useExtrinsicGuess=False,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        print(f"Failed to solve PnP for {image_path}")
        return None, None
    
    # Convert rotation vector to rotation matrix
    R_mat, _ = cv2.Rodrigues(rvec)
    
    # Build camera-to-world matrix (4x4)
    cam2world = np.eye(4)
    cam2world[:3, :3] = R_mat.T
    cam2world[:3, 3] = -R_mat.T @ tvec.flatten()
    
    if verbose:
        print(f"\nImage: {image_path}")
        print(f"Resolution: {width}x{height}")
        print(f"Focal length: {focal_length}")
        print(f"Camera-to-world matrix:\n{cam2world}")
    
    return cam2world, camera_matrix


def create_dataset_json(image_dir, output_path, focal_length=600, verbose=False):
    """
    Create dataset.json with camera parameters for all images in a directory.
    
    Args:
        image_dir: Directory containing images
        output_path: Path to save dataset.json
        focal_length: Focal length in pixels
        verbose: Print debug info
    """
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG'}
    image_paths = sorted([
        p for p in Path(image_dir).glob('*')
        if p.suffix in image_extensions
    ])
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    
    labels_list = []
    
    for image_path in tqdm(image_paths, desc="Estimating camera parameters"):
        cam2world, intrinsics = estimate_camera_params(
            str(image_path),
            focal_length=focal_length,
            verbose=verbose
        )
        
        if cam2world is None:
            print(f"Skipping {image_path.name}")
            continue
        
        # Flatten camera matrix and intrinsics
        cam2world_flat = cam2world[:3, :].flatten().tolist()  # 3x4 = 12 values
        intrinsics_flat = intrinsics.flatten().tolist()[:9]   # 3x3 = 9 values
        
        # Combine: 12 (pose) + 9 (intrinsics) + 4 (padding) = 25 values
        params = cam2world_flat + intrinsics_flat + [0, 0, 0, 0]
        
        labels_list.append([
            image_path.name,
            [str(p) for p in params]  # Convert to strings for JSON
        ])
    
    # Save to JSON
    output_dict = {"labels": labels_list}
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\nSaved dataset.json with {len(labels_list)} images to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate camera parameters for face images")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing images")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for dataset.json (default: <image_dir>/dataset.json)")
    parser.add_argument("--focal_length", type=float, default=600,
                        help="Approximate focal length in pixels")
    parser.add_argument("--verbose", action="store_true",
                        help="Print debug information")
    
    args = parser.parse_args()
    
    output_path = args.output or os.path.join(args.image_dir, "dataset.json")
    
    create_dataset_json(
        args.image_dir,
        output_path,
        focal_length=args.focal_length,
        verbose=args.verbose
    )
