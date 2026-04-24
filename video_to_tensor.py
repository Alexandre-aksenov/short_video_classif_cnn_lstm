import cv2
import argparse
import torch
import numpy as np
from typing import Sequence, Optional
from retinaface_inference import RetinaFaceDetector, cfg_mnet, cfg_re50


def video_to_tensor(video_path: str, detector: RetinaFaceDetector,
                    vis_thres: float = 0.5,
                    target_size: tuple[int, int] = (50, 50),
                    frames_to_extract: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    Processes a video and returns a 4D numpy tensor of shape (time, height, width, 3).
    
    Args:
        video_path: Path to the input video file.
        detector: An instance of RetinaFaceDetector.
        vis_thres: Confidence threshold for showing a face.
        target_size: The (height, width) to which each frame is resized.
        frames_to_extract: Optional list of frame indices to extract.
        
    Returns:
        A numpy array of shape (N, target_height, target_width, 3)
        where N is the number of frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return np.array([])

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frames_to_extract is None:
        indices = range(num_frames)
    else:
        indices = frames_to_extract

    frames_list = []
    warning_issued = False
    count = 0
    
    for idx in indices:
        frame = None
        if 0 <= idx < num_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = None
        
        if frame is None:
            if not warning_issued:
                print(f"Warning: Frame index {idx} out of bounds or could not be read. Producing black frames for invalid indices.")
                warning_issued = True
            # Produce a black frame
            processed_frame = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        else:
            # Detect the most confident face (returns 1x15 or 0x15)
            dets = detector.detect(frame)
            # Preprocess: crop to square around face, draw landmarks (except nose), resize to target_size
            processed_frame = detector.preprocess_face(frame, dets, vis_thres, target_size=target_size)
        
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
    parser.add_argument('--frames', type=int, nargs='+', default=None, help='List of frame indices to extract')
    
    args = parser.parse_args()

    # Disable gradients for inference
    torch.set_grad_enabled(False)
    
    # Load configuration
    cfg = cfg_mnet if args.network == "mobile0.25" else cfg_re50
    
    # Initialize detector from standalone library
    detector = RetinaFaceDetector(cfg, args.trained_model, args.cpu)
    
    print(f"Starting processing: {args.input}")
    tensor = video_to_tensor(args.input, detector, vis_thres=args.vis_thres, frames_to_extract=args.frames)
    
    if tensor.size > 0:
        print(f"Final tensor shape: {tensor.shape}")
        np.save(args.output, tensor)
        print(f"Tensor saved to {args.output}")
    else:
        print("No frames were processed.")
