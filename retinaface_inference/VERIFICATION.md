### Standalone RetinaFace Inference - Verification Steps

To ensure that the inference library is working correctly after any modification, the following verification steps can be performed:

#### 1. Linting and Static Analysis
Verify the syntax and type annotations of the `detector.py` file to catch early errors (like missing imports or incorrect type usage).
- **Tool**: You can use tools like `pylint`, `flake8`, or the internal IDE features (e.g., PyCharm's "Inspect Code").
- **Goal**: Ensure no syntax errors and that all intermediate variable types are correctly inferred.

#### 2. Shape Verification (Tensor Flow)
The `detect` method involves several transformations. You can verify the data flow by inserting print statements or using a debugger to check the shapes of the following variables:
- `img_raw`: `(H, W, 3)` (Original image)
- `img_torch`: `(1, 3, H, W)` (Input to the neural network)
- `loc`: `(1, N, 4)` (Encoded bounding boxes, where $N$ is the number of priors)
- `conf`: `(1, N, 2)` (Confidence scores for background/face)
- `landms_t`: `(1, N, 10)` (Encoded facial landmarks)
- `dets`: `(1, 15)` (Final result: 4 box coords + 1 score + 10 landmark coords)

#### 3. Execution Check
Run the standalone augmentation script to verify the end-to-end video processing:
```bash
python augment_video_standalone.py --input path/to/video.avi --output result.mp4
```
- **Expected Outcome**: The output video should be 50x50 and contain a cropped face with 4 landmark points (eyes, mouth corners) drawn in white circles.

#### 4. Confidence Filtering
Verify that `vis_thres` correctly filters detections:
- If a face is present but `dets` returns a resized original frame, check if the confidence score `b[4]` is below the `vis_thres` (default 0.5).

### 4. Video-to-Tensor Workflow
To verify the new output format (4D NumPy tensor), use the `video_to_tensor.py` script:
```bash
python video_to_tensor.py --input path/to/video.mp4 --output result.npy
```
After execution, verify the shape and data type:
```python
import numpy as np
data = np.load('result.npy')
print(data.shape) # Expected: (T, 50, 50, 3)
```
