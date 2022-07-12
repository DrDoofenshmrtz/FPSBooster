import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import cv2
from skimage.transform import resize

import torch
from torch.utils.data import Dataset


def obtain_frames_from_video(path, resize_to = (24, 32)):
    frames = []
    v_cap = cv2.VideoCapture(path)
    success = True
    
    while success:
        try:
            success, image = v_cap.read()
            image = resize(image, resize_to)
            frames.append(image)
        except Exception as e:
            pass

    frames = np.array(frames, dtype = np.float32)
    
    return frames

def sample_three_frames(arr_frames):
    # first frame is a random frame chosen between 0 and middle of the array
    first_frame_index = np.random.randint(0, arr_frames.shape[0] // 2)
    # third frame is a random frame chosen between middle of the array and its end
    third_frame_index = np.random.randint(arr_frames.shape[0] // 2, arr_frames.shape[0])
    # second frame is the frame which is mean to both the first and third frames. This ensures 
    # evenly spaced sampling
    second_frame_index = (first_frame_index + third_frame_index) // 2
    # slicing out the threee selected frames
    frame_indices = [first_frame_index, second_frame_index, third_frame_index]
    sampled_frames = arr_frames[frame_indices, :, :, :] 

    return sampled_frames

def display_frames(frames, resize_to = (240, 320)):
    fig = plt.figure(figsize = (30, 15))
    frames = [resize(frames[i], resize_to) for i in range(frames.shape[0])]
    for i, frame in enumerate(frames):
        ax = plt.subplot(1, 3, i + 1)
        plt.axis('off')
        plt.imshow(frames[i])
        
class UCF101Dataset(Dataset):
    def __init__(self, annotations_path, 
                 root_dir, 
                 convert_to_tensor = True):
    
        self.annotations_file = pd.read_csv(annotations_path, sep = " ", header = None, names = ['path', 'class'])
        self.root_dir = root_dir
        self.convert_to_tensor = convert_to_tensor

    def __len__(self):
        return len(self.annotations_file)    
        
    def __getitem__(self, idx):
        # obtaining path of the video indexed at 'idx'
        video_path = os.path.join(self.root_dir,
                                  self.annotations_file.iloc[idx, 0])
        frames = obtain_frames_from_video(video_path)
        sampled_frames = sample_three_frames(frames)
        
        if(self.convert_to_tensor == True):
            sampled_frames = torch.from_numpy(sampled_frames)
            sampled_frames = sampled_frames.permute(0, 3, 1, 2)
        
        return sampled_frames