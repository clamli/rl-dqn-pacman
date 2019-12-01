import torch
import config
import numpy as np
from gameAPI.game import GamePacmanAgent
agent = GamePacmanAgent(config)
frame, is_win, is_gameover, reward, action = agent.nextFrame(action=None)
frames = []
frames.append(frame)
frames.append(frame[:][:][:])

image = np.concatenate(frames, -1)
print(f'frame shape: {frame.shape}')
print(f'image shape: {image.shape}')

image_input = image.astype(np.float32) / 255.
image_input.resize((1, *image_input.shape))
print(f'image_input shape: {image_input.shape}')
input_tensor = torch.from_numpy(image_input).permute(0, 3, 1, 2).type(torch.FloatTensor)
print(input_tensor.shape)
#image_input_torch = torch.from_numpy(image_input).permute(0, 3, 1, 2).type(FloatTensor)

# print(frame.shape)
