# Classification of short videos

Although regression is the most common task in time series,
classification may also become useful in some applications. For instance, for flagging potentially inappropriate content in social media to prevent such videos from being posted online.

Videos are considered as time series of frames.

Therefore, it is appropriate to use  **Convolutional Neural Networks** (fore processing each frame) together with **Recurrent ones** to us the dependency between consecutive frames.


We use hybrid architectures called CNN-LSTM. The corresponding layer type is called **ConvLSTM** in Keras. 


# Data of the present project.

The dataset of this example is a subset of [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) by the Research Center in Computer Vision of the University of Central Florida. Our dataset is reduced the categories 'Apply Eye Makeup' and 'Apply Lipstick'.

The videos are preprocessed using the following steps:
* subsample frames to take one frame out of 6. In comparison to the frame rate of 25 frames/second, this leads to 4 frames per second;
* take only the 15 frames per video. This parameter has been determined from the minimal length of the videos (90 frames);
* Apply the pretrained model [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) to detect the face and facial landmarks in each frame.
* draw white circles on facial landmarks (eyes and corners of the mouth);
* crop each frame to the bounding box around the patient's face;
* resize each bounding box to the size 50x50 pixels;

The pretrained model has been modified to output only the most confident bounding box,
and to keep only inference using the model ResNet50. The new version is located in the folder `retinaface_inference`.


# Known issues and possible improvement.

The minority class is harder to predict.

The model can be improved using these ideas:
<ul>
    <li>Increase the new size.
    <li>use a second pretrained model for video segmentation to detect the brush.
    <li>increase the length of training (for this model is still learning at the epoch number 10).
    <li>preprocess the videos using Optical Flow.
    <li>check for overfitting, and whether the number of frames can be further reduced, for 15 frames in 3 sec may be excessive.</li>
</ul>

# Feedback and additional questions.

All questions about the source code should be addressed to its author Alexandre Aksenov:
* GitHub: Alexandre-aksenov
* Email: alexander1aksenov@gmail.com
