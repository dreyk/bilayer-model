import argparse
import cv2
from infer import InferenceWrapper
import numpy as np

args_dict = {
    'project_dir': './models/bilayer_model',
    'init_experiment_dir': './models/bilayer_model/runs/vc2-hq_adrianb_paper_main',
    'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'init_which_epoch': '2225',
    'num_gpus': 0,
    'experiment_name': 'vc2-hq_adrianb_paper_enhancer',
    'which_epoch': '1225',
    'spn_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, texture_enhancer',
    'enh_apply_masks': False,
    'inf_apply_masks': False}

# Initialization
module = InferenceWrapper(args_dict)

src = cv2.imread('./test_data/44-china.jpg')
trg = cv2.imread('./test_data/google2.jpg')
src = np.expand_dims(src,axis=0)
# Input data for intiialization and inference
data_dict = {
    'source_imgs': trg,
    'target_imgs': src
}

def to_image(img_tensor, seg_tensor=None):
    img_array = ((img_tensor.clamp(-1, 1).cpu().numpy() + 1) / 2).transpose(1, 2, 0) * 255
    
    if seg_tensor is not None:
        seg_array = seg_tensor.cpu().numpy().transpose(1, 2, 0)
        img_array = img_array * seg_array + 255. * (1 - seg_array)

    return img_array.astype(np.uint8)

# Inference
data_dict = module(data_dict)

# Outputs (images are in [-1, 1] range, segmentation masks -- in [0, 1])
imgs = data_dict['pred_enh_target_imgs']
segs = data_dict['pred_target_segs']
pred_img = to_image(imgs[0, 0], segs[0, 0])
cv2.imwrite('result.jpg',pred_img)

