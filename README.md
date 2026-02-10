# Classification of short videos

Although regression is the most common task in time series,
classification may also become useful in some applications. For instance, for flagging potentially inappropriate content in social media to prevent such videos from being posted online.

Videos are considered as time series of frames.

Therefore, it is appropriate to use  **Convolutional Neural Networks** (fore processing each frame) together with **Recurrent ones** to us the dependency between consecutive frames.


We use hybrid architectures called CNN-LSTM. The corresponding layer type is called **ConvLSTM** in Keras. 


# Data of the present project.

The dataset of this example is is a subset of [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) by the Research Center in Computer Vision of the University of Central Florida. Our dataset is reduced the categories 'Apply Eye Makeup' and 'Apply Lipstick'.

A small number of images (15) is used per video. This should be compared with the frame rate:  25 images per second. Our simplified version allows reducing the data size and time of computation.

# Known issues and possible improvement.

The minority class is harder to predict, but precision/recall are quite balanced.

The model can be improved using these ideas:
<ul>
    <li>increase the length of training (for this model is still learning at the epoch number 10).
    <li>preprocess the videos using Optical Flow.
    <li>check for overfitting, and whether the number of frames can be reduced, for 15 frames in 3 sec may be excessive.</li>
</ul>

# Feedback and additional questions.

All questions about the source code should be addressed to its author Alexandre Aksenov:
* GitHub: Alexandre-aksenov
* Email: alexander1aksenov@gmail.com
