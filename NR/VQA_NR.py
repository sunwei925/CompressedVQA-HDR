import argparse
import os
import sys
import numpy as np
import torch
import NR_model
import cv2
from PIL import Image
from torchvision import transforms
import random

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def video_processing(dist):

    video_name = dist

    video_capture = cv2.VideoCapture()
    video_capture.open(video_name)
    cap=cv2.VideoCapture(video_name)

    video_channel = 3

    size = 384

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # the heigh of frames
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # the width of frames
    
    if video_height > video_width:
        video_width_resize = size
        video_height_resize = int(video_width_resize/video_width*video_height)
    else:
        video_height_resize = size
        video_width_resize = int(video_height_resize/video_height*video_width)
        
    dim = (video_width_resize, video_height_resize)

    video_length_read = int(video_length/video_frame_rate)

    transformations = transforms.Compose([transforms.Resize(size),transforms.CenterCrop(size),transforms.ToTensor(),\
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    transformed_video = torch.zeros([video_length_read, video_channel,  size, size])

    video_read_index = 0
    frame_idx = 0
            
    for i in range(video_length):
        has_frames, frame = video_capture.read()
        if has_frames:

            # key frame
            if (video_read_index < video_length) and (frame_idx % (int(video_frame_rate)) == int(video_frame_rate / 2)):
                read_frame = cv2.resize(frame, dim)
                read_frame = Image.fromarray(cv2.cvtColor(read_frame,cv2.COLOR_BGR2RGB))
                read_frame = transformations(read_frame)
                transformed_video[video_read_index] = read_frame
                video_read_index += 1

            frame_idx += 1

    if video_read_index < video_length_read:
        for i in range(video_read_index, video_length_read):
            transformed_video[i] = transformed_video[video_read_index - 1]

    video_capture.release()


    return transformed_video, video_name

def main():
    parser = argparse.ArgumentParser(description='No Reference HDR Video Quality Assessment')
    parser.add_argument('--distorted', type=str, required=True, help='Path to the distorted video')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--profile_path', type=str, required=True, help='Path to the profile file (.npy file)')
    args = parser.parse_args()

    seed = 2
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load profile and model
    profile = np.load(args.profile_path)
    print("Loading profile from:", args.profile_path)
    print(profile)
    
    model = NR_model.SigLIP2_384_multi_dataset()
    model = model.to(device)
    print('Using the pretrained model from:', args.model_path)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    video_dist, video_name = video_processing(args.distorted)

    with torch.no_grad():
        model.eval()      

        video_dist = video_dist.to(device)
        video_dist = video_dist.unsqueeze(dim=0)

        output, _, _, _, _ = model(video_dist)       
        y_val = output.item()
        y_val = logistic_func(y_val, *profile)

        print('Video name: ' + video_name)
        print('Quality score: {:.4f}'.format(y_val))

if __name__ == '__main__':
    main()