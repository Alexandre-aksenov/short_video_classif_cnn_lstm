# Module for importing data into the model.

import numpy as np
import cv2
import os

from collections.abc import Sequence


def subsample_frames(seq_len: int, period: int, residue=0) -> list[int]:
    """ the list of frames to be extracted for given parameters

    Args:
        seq_len (int): the number of frames that will enter the dataset
        period (int): periodicity
        residue (int, default=0):
            the residue of frame passed to model modulo the period,
            unlikely to be important here.

    Returns:
        list[int]: subsampled list of indices
    """
    residue = residue % period 
    
    frames_to_consider = seq_len * period
    res = list(range(residue, frames_to_consider, period))

    # check length of result and return
    assert len(res) == seq_len
    return res


def frames_extraction(video_path: str,
                        img_width:int,
                        img_height:int,
                        frames_to_extract: Sequence[int]) -> list:
    """ Get frames from a video.
    
    Args:
        video_path (str): path to video
    
    Input global variables:
        img_width, img_height (int)
        frames_to_extract (list or range of int)
    
    Raises:
        EOFError: video is shorter than the required number of frames
            equal frames_to_extract[-1] + 1

    Returns:
        list[np.ndarray]: video converted to list of 3d tensors (N_width, N_height, N_channel).
        
    Returns frames with periodic sampling.
    """
    
    frames_list = []

    vidObj = cv2.VideoCapture(video_path)
    # An instance VideoCapture
    # https://www.scaler.com/topics/cv2-videocapture/

    for n_frame in range(frames_to_extract[-1] + 1):
    
        success, image = vidObj.read() # Read images 1-by-1
        
        if success and (n_frame in frames_to_extract):
            
            # Using OpenCV, reshape the image 
            img_reshaped: np.ndarray = cv2.resize(image, dsize=(img_width, img_height))
            
            # Append the frame to the list
            frames_list.append(img_reshaped)
            
        elif not success:
            # Print error messsage and exit the loop
            print(f"Failed to read the frame {n_frame} from the file {video_path}")
            # the video is too short,
            # the condition "if len(frames) == seq_len" will remove it from dataset

    return frames_list


def create_dataset(input_dir: str,
                    classes: Sequence[str],
                    img_width:int,
                    img_height:int,
                    # frames_to_extract: Sequence[int], ->
                    period: int, 
                    seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Write an array with all images of each video
    and the corresponding labels.

    Input:
        input_dir (str): path to folder with the videos.
        classes: Sequence[str]
        img_width, img_height: int
        frames_to_extract (range or list of int)
        seq_len: int
    
    Returns formatted dataset:
        X (np.ndarray : (N_samples, N_timesteps, N_width, N_height, N_channel));
        Y (np.ndarray : (N_samples, N_classes))
    
    This function can be sped up by pre-allocating numpy arrays.
    """
    frames_to_extract = subsample_frames(seq_len, period)
    
    X = []
    Y = []

    for c in classes:
        print(c)
        files_list = os.listdir(os.path.join(input_dir, c))
        # list[str] : list of file names in the folder.
        # It's assumed the following loop that all files are videos.
        
        for f in files_list:
            frames = frames_extraction(os.path.join(os.path.join(input_dir, c), f), img_width, img_height, frames_to_extract)
            # frames: list of frames.
            
            if len(frames) == seq_len:
                X.append(frames)

                y = [0]*len(classes)
                y[classes.index(c)] = 1
                # list.index :
                # https://docs.python.org/3/tutorial/datastructures.html#more-on-lists
                
                Y.append(y)
                # y (list of 0es and 1es of length len(classes)):
                # the one-hot encoding of the index of file.

    # convert X, Y to tensor and matrix respectively.
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y
