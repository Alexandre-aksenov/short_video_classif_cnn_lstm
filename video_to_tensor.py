import cv2
import argparse
import torch
import numpy as np
import os
from retinaface_inference import RetinaFaceDetector, cfg_mnet, cfg_re50

def video_to_tensor(video_path: str, detector: RetinaFaceDetector, vis_thres: float = 0.5, target_size: tuple = (50, 50)) -> np.ndarray:
    """
    Processes a video and returns a 4D numpy tensor of shape (time, height, width, 3).
    
    Args:
        video_path: Path to the input video file.
        detector: An instance of RetinaFaceDetector.
        vis_thres: Confidence threshold for showing a face.
        target_size: The (height, width) to which each frame is resized.
        
    Returns:
        A numpy array of shape (N, target_height, target_width, 3) where N is the number of frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return np.array([])

    frames_list = []
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect the most confident face (returns 1x15 or 0x15)
        dets = detector.detect(frame)
        
        # Preprocess: crop to square around face, draw landmarks (except nose), resize to target_size
        # If no face is detected, it simply resizes the original frame to target_size.
        processed_frame = detector.preprocess_face(frame, dets, vis_thres, target_size=target_size)
        
        # processed_frame is BGR (from OpenCV). For most classifiers, you might want RGB.
        # However, following the current 'preprocess_face' logic which uses cv2, we keep BGR.
        frames_list.append(processed_frame)
        
        count += 1
        if count % 10 == 0:
            print(f"Processed {count} frames...")

    cap.release()
    
    # Convert list of arrays to a single 4D numpy tensor
    if len(frames_list) > 0:
        tensor = np.stack(frames_list, axis=0)
        return tensor
    else:
        return np.array([])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface Video to Tensor Preprocessing')
    parser.add_argument('-m', '--trained_model', default='./all_weights/mobilenet0.25_Final.pth',
                        type=str, help='Trained state_dict file path')
    parser.add_argument('--network', default='mobile0.25', help='mobile0.25 or resnet50')
    parser.add_argument('--input', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='output_tensor.npy', help='Path to save the output .npy tensor')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--vis_thres', type=float, default=0.5, help='Visualization threshold')
    
    args = parser.parse_args()

    # Disable gradients for inference
    torch.set_grad_enabled(False)
    
    # Load configuration
    cfg = cfg_mnet if args.network == "mobile0.25" else cfg_re50
    
    # Initialize detector from standalone library
    detector = RetinaFaceDetector(cfg, args.trained_model, args.cpu)
    
    print(f"Starting processing: {args.input}")
    tensor = video_to_tensor(args.input, detector, vis_thres=args.vis_thres)
    
    if tensor.size > 0:
        print(f"Final tensor shape: {tensor.shape}")
        np.save(args.output, tensor)
        print(f"Tensor saved to {args.output}")
    else:
        print("No frames were processed.")
