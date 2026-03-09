import numpy as np
import cv2
import os


def frames_extraction(video_path: str, img_width, img_height, frames_to_extract) -> list:
    """ Get frames from a video.
    
    Args:
        video_path (str): path to video
    
    Input global variables:
        img_width, img_height (int)
        frames_to_extract (range or list of int)
    
    Raises:
        EOFError: video is shorter than 'seq_len' frames

    Returns:
        list[np.ndarray ?]: video converted to 4d tensor
        
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
            raise EOFError

    return frames_list


def create_dataset(input_dir: str, classes, img_width, img_height, frames_to_extract, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Write an array with all images of each video
    and the corresponding labels.

    Input:
        input_dir (str): path to folder with the videos.
        seq_len: int

    Input global variable: classes: iterable[str]
    Input global variables for 'frames_extraction':
        img_width, img_height, seq_len (int)
    
    Returns formatted dataset:
        X (np.ndarray : (N_samples, N_timesteps, N_width, N_height, N_channel));
        Y (np.ndarray : (N_samples, N_classes))
    
    This function can be sped up by pre-allocating numpy arrays.
    """
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
