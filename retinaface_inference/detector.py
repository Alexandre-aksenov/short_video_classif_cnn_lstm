import torch
import numpy as np
import cv2
from typing import Tuple
from .layers.functions.prior_box import PriorBox
from .models.retinaface import RetinaFace
from .utils.box_utils import decode, decode_landm
from .utils.timer import Timer

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class RetinaFaceDetector:
    def __init__(self, cfg, trained_model, cpu=False):
        self.cfg = cfg
        self.device = torch.device("cpu" if cpu else "cuda")
        self.net = RetinaFace(cfg=cfg, phase='test')
        self.net = load_model(self.net, trained_model, cpu)
        self.net.eval()
        self.net = self.net.to(self.device)
        self.timer = {'forward_pass': Timer(), 'misc': Timer()}

    def detect(self, img_raw: np.ndarray, nms_threshold: float = 0.4) -> np.ndarray:
        # img_raw shape: H x W x 3
        img: np.ndarray = np.float32(img_raw)  # shape: H x W x 3
        assert img.ndim == 3 and img.shape[2] == 3, f"Unexpected img shape: {img.shape}"
        im_height: int
        im_width: int
        im_height, im_width, _ = img.shape
        # scale shape: [4]
        scale: torch.Tensor = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        assert scale.shape == (4,), f"Unexpected scale shape: {scale.shape}"
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)  # shape: 3 x H x W
        assert img.shape == (3, im_height, im_width), f"Unexpected transposed img shape: {img.shape}"
        img_torch: torch.Tensor = torch.from_numpy(img).unsqueeze(0)  # shape: 1 x 3 x H x W
        assert img_torch.shape == (1, 3, im_height, im_width), f"Unexpected img_torch shape: {img_torch.shape}"
        img_torch = img_torch.to(self.device)
        scale = scale.to(self.device)

        self.timer['forward_pass'].tic()
        loc: torch.Tensor
        conf: torch.Tensor
        landms_t: torch.Tensor
        # loc shape: 1 x N x 4, conf shape: 1 x N x 2, landms_t shape: 1 x N x 10
        # where N is the number of prior boxes
        loc, conf, landms_t = self.net(img_torch)  # forward pass
        num_priors = loc.shape[1]
        assert loc.shape == (1, num_priors, 4), f"Unexpected loc shape: {loc.shape}"
        assert conf.shape == (1, num_priors, 2), f"Unexpected conf shape: {conf.shape}"
        assert landms_t.shape == (1, num_priors, 10), f"Unexpected landms_t shape: {landms_t.shape}"
        self.timer['forward_pass'].toc()

        self.timer['misc'].tic()
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        # priors shape: N x 4
        priors: torch.Tensor = priorbox.forward()
        assert priors.shape == (num_priors, 4), f"Unexpected priors shape: {priors.shape}"
        priors = priors.to(self.device)
        prior_data: torch.Tensor = priors.data
        # boxes_t shape: N x 4
        boxes_t: torch.Tensor = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        assert boxes_t.shape == (num_priors, 4), f"Unexpected boxes_t shape: {boxes_t.shape}"
        # boxes shape: N x 4
        boxes: np.ndarray = (boxes_t * scale).cpu().numpy()
        assert boxes.shape == (num_priors, 4), f"Unexpected boxes shape: {boxes.shape}"
        # scores shape: N
        scores: np.ndarray = conf.squeeze(0).data.cpu().numpy()[:, 1]
        assert scores.shape == (num_priors,), f"Unexpected scores shape: {scores.shape}"
        # landms_t_dec shape: N x 10
        landms_t_dec: torch.Tensor = decode_landm(landms_t.data.squeeze(0), prior_data, self.cfg['variance'])
        assert landms_t_dec.shape == (num_priors, 10), f"Unexpected landms_t_dec shape: {landms_t_dec.shape}"
        # scale1 shape: [10]
        scale1: torch.Tensor = torch.Tensor([img_torch.shape[3], img_torch.shape[2], img_torch.shape[3], img_torch.shape[2],
                               img_torch.shape[3], img_torch.shape[2], img_torch.shape[3], img_torch.shape[2],
                               img_torch.shape[3], img_torch.shape[2]])
        assert scale1.shape == (10,), f"Unexpected scale1 shape: {scale1.shape}"
        scale1 = scale1.to(self.device)
        # landms shape: N x 10
        landms: np.ndarray = (landms_t_dec * scale1).cpu().numpy()
        assert landms.shape == (num_priors, 10), f"Unexpected landms shape: {landms.shape}"

        # keep top-1 (NMS is redundant for single face selection)
        order = scores.argsort()[::-1]
        top_idx = order[:1]
        boxes = boxes[top_idx]  # shape: 1 x 4 (or 0 x 4 if N=0)
        landms = landms[top_idx]  # shape: 1 x 10 (or 0 x 10 if N=0)
        scores = scores[top_idx]  # shape: 1 (or 0 if N=0)
        num_final = 1 if num_priors > 0 else 0
        assert boxes.shape == (num_final, 4)
        assert landms.shape == (num_final, 10)
        assert scores.shape == (num_final,)

        # dets shape: 1 x 5 (or 0 x 5)
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        assert dets.shape == (num_final, 5)
        # final dets shape: 1 x 15 (or 0 x 15)
        dets = np.concatenate((dets, landms), axis=1)
        assert dets.shape == (num_final, 15)
        self.timer['misc'].toc()

        return dets

    def draw_on_image(self, img_raw, dets, vis_thres=0.5):
        for b in dets:
            if b[4] < vis_thres:  # Skip detections with low confidence (b[4])
                continue
            # text = "{:.4f}".format(b[4])  # rm text on image -> comment out
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            # cx = b[0]
            # cy = b[1] + 12  # rm text on image -> comment out +12

            # Init version: write the confidence level on top of rectangle
            # rm text on image -> comment out
            # cv2.putText(img_raw, text, (cx, cy),
            #            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        return img_raw

    def preprocess_face(self, img_raw: np.ndarray, dets: np.ndarray, vis_thres: float = 0.5, target_size: Tuple[int, int] = (50, 50)) -> np.ndarray:
        if len(dets) == 0 or dets[0][4] < vis_thres:
            return cv2.resize(img_raw, target_size)
        
        # Take the most confident detection
        b: np.ndarray = dets[0]
            
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, b[:4])
        w, h = x2 - x1, y2 - y1
        
        # Center of the face
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Make it a square crop (side length = max(w, h))
        side: int = max(w, h)
        half_side: int = side // 2
        
        # Define crop boundaries (ensure they are within image limits)
        img_h: int
        img_w: int
        img_h, img_w = img_raw.shape[:2]
        ny1: int = max(0, cy - half_side)
        ny2: int = min(img_h, cy + half_side)
        nx1: int = max(0, cx - half_side)
        nx2: int = min(img_w, cx + half_side)
        
        # Crop the image
        cropped: np.ndarray = img_raw[ny1:ny2, nx1:nx2].copy()
        
        # Draw 4 circles at landmarks (excluding the nose at b[9:11])
        # Note: landmark coordinates must be shifted to match the cropped image
        landmark_indices = [(5, 6), (7, 8), (11, 12), (13, 14)]
        for lx_idx, ly_idx in landmark_indices:
            lx: int = int(b[lx_idx])
            ly: int = int(b[ly_idx])
            # Shift landmarks by the crop top-left corner
            slx, sly = lx - nx1, ly - ny1
            cv2.circle(cropped, (slx, sly), 1, (255, 255, 255), 2)
            
        # Resize to fixed size (e.g., 50x50)
        resized: np.ndarray = cv2.resize(cropped, target_size)
        return resized
