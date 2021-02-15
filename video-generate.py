import argparse
import cv2
from infer_land import InferenceWrapper
import numpy as np
import logging
import subprocess
import torch
import face_alignment
import random
import math
import transform3d.euler as euler
    
def to_image(img_tensor, seg_tensor=None):
    img_array = ((img_tensor.clamp(-1, 1).cpu().numpy() + 1) / 2).transpose(1, 2, 0) * 255
    
    if seg_tensor is not None:
        seg_array = seg_tensor.cpu().numpy().transpose(1, 2, 0)
        img_array = img_array * seg_array + 255. * (1 - seg_array)

    return img_array.astype(np.uint8)
    
def process(args):
    num_gpus = 1 if torch.cuda.is_available() else 0
    args_dict = {
        'project_dir': args.project,
        'init_experiment_dir': args.project+'/runs/vc2-hq_adrianb_paper_main',
        'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
        'init_which_epoch': '2225',
        'num_gpus': num_gpus,
        'experiment_name': 'vc2-hq_adrianb_paper_enhancer',
        'which_epoch': '1225',
        'spn_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, texture_enhancer',
        'enh_apply_masks': False,
        'inf_apply_masks': False}

    logging.info("Loading nets")
    # Initialization
    module = InferenceWrapper(args_dict)

    logging.info("Prepare data")
    
    trg = cv2.imread(args.puppet)
    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    width = 256
    height = 256
    vout = cv2.VideoWriter(args.out_video+'tmp.mp4', fourcc, fps, (width,height))
    
    count = 0
    prev_percent = 0
    frame_count = int(fps*args.record_time)
    
    (source_poses, 
         source_imgs, 
         source_segs, 
         source_stickmen) = module.preprocess_data(trg, True)
    data_dict = {
            'source_imgs': source_imgs,
            'source_poses': source_poses,
            'source_segs':source_segs,
            'source_stickmen':source_stickmen,
            }
    use_device = 'cpu' if num_gpus<1 else 'cuda'
    fa3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device=use_device)
    src_img = to_image(source_imgs[0][0])
    source_land = source_poses[0][0].cpu().numpy()
    source_land = np.reshape(source_land,(68,2))
    detected_faces = fa3d.face_detector.detect_from_image(src_img[:, :, ::-1].copy())
    r = detected_faces[0]
    scale = (r[2] - r[0] + r[3] - r[1]) / 195
    res = fa3d.get_landmarks(src_img, detected_faces=[r])[0]
    landmark_first = []
    for i in range(res.shape[0]):
        landmark_first.append([source_land[i, 0], source_land[i, 1], res[i, 2] / (200 * scale)])
    landmark_first = np.array(landmark_first, dtype=np.float32)
    brow_amp = np.max(np.abs(landmark_first[[37, 38], 1] - landmark_first[[41, 40], 1]))
    last_motion = 0
    a1_speed = (random.randint(0, 200) / 100 - 0)/10
    a2_speed = (random.randint(0, 200) / 100 - 0)/10
    next_blink = random.randint(5, 10)
    blink_duration = random.randint(2, 20) / 50
    next_brow = random.randint(5, 15)
    brow_duration = random.randint(4, 30) / 50
    for count in range(frame_count):
        t = count/fps
        p = int(count*100/frame_count)
        if p !=prev_percent:
            logging.info("Process: %d",p)
            prev_percent = p
        start = t
        next_landmark = landmark_first.copy()
        if start > next_blink + blink_duration:
            blink_duration = random.randint(2, 20) / 50
            next_blink = start + random.randint(5, 15)
        elif start > next_blink:
            next_landmark[[37, 38], 1] = next_landmark[[41, 40], 1]
            next_landmark[[43, 44], 1] = next_landmark[[47, 46], 1]

        if brow_amp > 0:
            if start > next_brow + brow_duration:
                next_brow = t + random.randint(5, 20)
                brow_duration = random.randint(2, 10) / 50
        elif start > next_brow:
            a = brow_amp * random.randint(5, 10) / 10
            next_landmark[17: 21, 1] = next_landmark[17: 21, 1] - a
            next_landmark[22: 26, 1] = next_landmark[22: 26, 1] - a
        if (t - last_motion > 5):
                a1_speed = (random.randint(0, 200) / 100 - 0)/5
                a2_speed = (random.randint(0, 200) / 100 - 0)/5
                last_motion = t
        a1 = 10 * math.pi / 180 * math.sin(t * a1_speed * 4 * math.pi)
        a2 = 10 * math.pi / 180 * math.sin(t * a2_speed * 4 * math.pi)
        a3 = 0
        m = euler.euler2mat(a1, a2, a3)
        next_landmark = np.dot(m, next_landmark.T).T
        poses = []
        next_landmark = next_landmark[:,0:2].astype(np.float32).copy()
        poses.append(torch.from_numpy(next_landmark).view(-1))
        poses = torch.stack(poses, 0)[None]
        if num_gpus>0:
            poses = poses.cuda()
        data_dict['target_poses'] =  poses
        data_dict = module(data_dict)

        # Outputs (images are in [-1, 1] range, segmentation masks -- in [0, 1])
        imgs = data_dict['pred_enh_target_imgs']
        segs = data_dict['pred_target_segs']
        pred_img = to_image(imgs[0, 0], segs[0, 0])
        vout.write(pred_img)
    vout.release()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )

    #model config
    parser.add_argument("--puppet", default='./test_data/google2.jpg')
    parser.add_argument(
        "--out_video",default="out.mp4", help="Out type"
    )
    parser.add_argument("--project", default='./models/bilayer_model', type=str)
    parser.add_argument("--record_time", default=0.5, type=float)
    args = parser.parse_args()
    process(args)