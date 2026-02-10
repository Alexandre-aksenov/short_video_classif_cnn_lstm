# Adapt RandomUnderSampler to work with tensors

import numpy as np
from imblearn.under_sampling import RandomUnderSampler


def to_matrix(X_tens: np.ndarray) -> np.ndarray:
    """
    Convert a tensor to a 2D matrix, by flattening all dimensions except the first one.
    """
    in_shape = X_tens.shape
    return X_tens.reshape(in_shape[0], -1)


def undersample_tensor(X_tens: np.ndarray, y_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Undersample the majority class in a dataset.
    
    Args:
        X_tens (np.ndarray): tensor of shape (N_samples, N_timesteps, N_width, N_height, N_channel)
        y_mat (np.ndarray): matrix of shape (N_samples, N_classes)
    
    Returns:
        tuple[np.ndarray, np.ndarray]: undersampled dataset
    """
    # Convert tensor to matrix
    X_mat = to_matrix(X_tens)

    # Undersample the majority class
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_mat, y_mat) # Pylance: tuple size mismatch

    # Convert back to tensor
    in_shape = X_tens.shape
    out_shape = (X_res.shape[0], in_shape[1], in_shape[2], in_shape[3], in_shape[4])
    X_res_tens = X_res.reshape(out_shape)

    return X_res_tens, y_res


# test to_matrix
if __name__ == "__main__":
    ex_tens = np.arange(2*3*4).reshape(2, 3, 4)
    print(ex_tens)
    
    print("----")
    print("It is reshaped to:")
    print(to_matrix(ex_tens))
