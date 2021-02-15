import argparse
import cv2
import numpy as np
import logging
import transform3d.euler as euler
import mel2land.audio as audio
import speech.landmark_norm as lnorm
import torch
import face_alignment

def resize(frame, mx, force=False):
    w = frame.shape[1]
    h = frame.shape[0]
    if w > h and (w > mx or force):
        h = int(h * mx / w)
        w = mx
        frame = cv2.resize(frame, (w, h))
    elif h > mx or force:
        w = int(w * mx / h)
        h = mx
        frame = cv2.resize(frame, (w, h))
    return frame

def process(args):
    num_gpus = 1 if torch.cuda.is_available() else 0
    use_device = 'cpu' if num_gpus<1 else 'cuda'
    fa3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device=use_device)
    vin = cv2.VideoCapture(args.video)
    fps = vin.get(cv2.CAP_PROP_FPS)
    landmark_first = None
    times = []
    angles = []
    count = 0
    while True:
        ret,frame = vin.read()
        if not ret:
            break
        orig_img = resize(frame, 1024)
        src_img = orig_img.copy()
        detected_faces = fa3d.face_detector.detect_from_image(src_img[:, :, ::-1].copy())
        r = detected_faces[0]
        scale = (r[2] - r[0] + r[3] - r[1]) / 195
        res = fa3d.get_landmarks(src_img.copy(), detected_faces=[r])[0]
        landmark = []
        for i in range(res.shape[0]):
            landmark.append([res[i, 0] / src_img.shape[1], res[i, 1] / src_img.shape[0], res[i, 2] / (200 * scale)])
        landmark = np.array(landmark, dtype=np.float32)
        if landmark_first is None:
            landmark_first = landmark
            times.append(0.0)
            angles.append([0.0,0.0,0.0])
            count += 1
            continue
        t = count/fps
        times.append(t)
        _, transform = lnorm.norm3d_t(landmark_first.copy(), landmark)
        a1, a2, a3 = euler.mat2euler(transform)
        logging.info("{}-{}-{}".format(a1,a2,a3))
        angles.append([a1,a2,a3])
        
    vin.release()
    np.savez(args.out+"/track",angles=np.array(angles,np.float32),times=np.array(times,np.float32))
    
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
    parser.add_argument("--video", default=None)
    parser.add_argument("--out",default=None)
    args = parser.parse_args()
    process(args)