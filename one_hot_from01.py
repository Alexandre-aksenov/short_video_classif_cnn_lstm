# Reencode a target vector of 0/1 to one-hot encoding.
# https://www.geeksforgeeks.org/numpy/how-to-convert-an-array-of-indices-to-one-hot-encoded-numpy-array/

import numpy as np
from sklearn.preprocessing import LabelBinarizer


binarizer = LabelBinarizer(sparse_output=False)  # to get a dense array.
binarizer.fit(range(3))


def onehot_from01(y_res: np.ndarray) -> np.ndarray:
    """
    # keep only the first 2 columns, corresponding to the 2 classes of our problem.

    Arg:
        y_res (1D np.ndarray): target vector of 0/1, of shape (N_samples,)

    Global input:
        binarizer (LabelBinarizer): binarizer fitted on 3 classes (0, 1, 2).

    Returns:
        np.ndarray: one-hot encoded target matrix of shape (N_samples, 2)
    
    PyLance raises a false positive: binarizer.transform "can" return an spmatrix.
    """
    # return binarizer.transform(y_res)[:, :2]
    return np.array(binarizer.transform(y_res))[:, :2]


if __name__ == "__main__":
    print(binarizer.classes_)  # [0, 1, 2]
    y_res = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    print("Original vector:")
    print(y_res)
    
    y_res_onehot = onehot_from01(y_res)
    print(y_res_onehot.shape)  # (10, 2)
    print(f"Vector converted to one-hot encoding:")
    print(y_res_onehot)
    """
    [[1 0]
    [0 1]
    [1 0]
    [0 1]
    [0 1]
    [1 0]
    [1 0]
    [0 1]
    [1 0]
    [0 1]]
    
    """
