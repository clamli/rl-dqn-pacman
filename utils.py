import numpy as np
import torch
import config
import pickle
from skimage.transform import resize


FloatTensor = torch.cuda.FloatTensor if config.use_cuda else torch.FloatTensor


def convert_idx_to_2_dim_tensor(action):
    if action == 0:
        return [-1, 0]
    elif action == 1:
        return [1, 0]
    elif action == 2:
        return [0, -1]
    elif action == 3:
        return [0, 1]


def convert_2_dim_tensor_to_4_dim_tensor(action):
    if action == [-1, 0]:
        return [1, 0, 0, 0]
    elif action == [1, 0]:
        return [0, 1, 0, 0]
    elif action == [0, -1]:
        return [0, 0, 1, 0]
    elif action == [0, 1]:
        return [0, 0, 0, 1]


def frames_to_tensor(frames):
    images_input = []
    for frame in frames:
        if config.use_simple:
            image = np.dot(frame[..., :3], [0.114, 0.587, 0.299])
            image = resize(image, (80, 80, 1))
        else:
            image = np.concatenate([frame], -1)

        image_input = image.astype(np.float32) / 255.
        image_input.resize((1, *image_input.shape))
        images_input.append(image_input)

    return torch.from_numpy(np.concatenate(images_input, 0)).permute(0, 3, 1, 2).type(FloatTensor)


def single_frame_to_tensor(frame):
    if config.use_simple:
        image = np.dot(frame[..., :3], [0.114, 0.587, 0.299])
        image = resize(image, (80, 80, 1))
    else:
        image = np.concatenate([frame], -1)
    image_input = image.astype(np.float32) / 255.
    image_input.resize((1, *image_input.shape))
    image_input_torch = torch.from_numpy(image_input).permute(0, 3, 1, 2).type(
        FloatTensor)

    return image_input_torch


def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
