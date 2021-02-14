import argparse
import cv2
from infer import InferenceWrapper
import numpy as np
import logging
import subprocess
import torch
def post_process_video(tmp_file, src_file, dest_file):
    command = [
        'ffmpeg',
        '-y',
        '-i', tmp_file,
        '-i', src_file,
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-c:a', 'aac',
        dest_file
    ]
    cmd = subprocess.Popen(command, stderr=subprocess.STDOUT)
    cmd.wait(60)
    
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
    vin = cv2.VideoCapture(args.video)

    frame_count = int(cv2.VideoCapture.get(vin, cv2.CAP_PROP_FRAME_COUNT))
    fps = vin.get(cv2.CAP_PROP_FPS)/3
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    width = 256
    height = 256
    vout = cv2.VideoWriter(args.out_video+'tmp.mp4', fourcc, fps, (width,height))
    
    count = 0
    prev_percent = 0
    while True:
        ret,frame = vin.read()
        if not ret:
            break
        if count % 3 != 0:
            count += 1
            continue
        count += 1
        p = int(count*100/frame_count)
        if p !=prev_percent:
            logging.info("Process: %d",p)
            prev_percent = p
        logging.info('Frame: {}'.format(count))
        src = np.expand_dims(frame,axis=0)
        # Input data for intiialization and inference
        data_dict = {
            'source_imgs': trg,
            'target_imgs': src
        }
        # Inference
        data_dict = module(data_dict)

        # Outputs (images are in [-1, 1] range, segmentation masks -- in [0, 1])
        imgs = data_dict['pred_enh_target_imgs']
        segs = data_dict['pred_target_segs']
        pred_img = to_image(imgs[0, 0], segs[0, 0])
        vout.write(pred_img)
    vin.release()
    vout.release()
    post_process_video(args.out_video+'tmp.mp4', args.video, args.out_video)

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
    parser.add_argument("--video", default='./test_data/BirdsShort.mp4', type=str)
    parser.add_argument(
        "--out_video",default="out.mp4", help="Out type"
    )
    parser.add_argument("--project", default='./models/bilayer_model', type=str)
    args = parser.parse_args()
    process(args)