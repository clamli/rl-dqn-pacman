import os
import torch

'''dqn'''
on_TACC = False
M = 10000
epsilon = 0.1
sample_size = 32
use_cuda = False
gamma = 0.95
start_training_threshold = 5000
save_model_threshold = 60000
max_memory_size = 100000
eps_start = 1.0
eps_end = 0.1
eps_num_steps = 10000

'''game'''
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)
SKYBLUE = (0, 191, 255)
layout_filepath = 'layouts/mediumClassic.lay' # decide the game map
ghost_image_paths = [(each.split('.')[0], os.path.join(os.getcwd(), each)) for each in ['gameAPI/images/Blinky.png', 'gameAPI/images/Inky.png', 'gameAPI/images/Pinky.png', 'gameAPI/images/Clyde.png']]
scaredghost_image_path = os.path.join(os.getcwd(), 'gameAPI/images/scared.png')
pacman_image_path = ('pacman', os.path.join(os.getcwd(), 'gameAPI/images/pacman.png'))
font_path = os.path.join(os.getcwd(), 'gameAPI/font/ALGER.TTF')
grid_size = 32
operator = 'ai' # 'person' or 'ai', used in demo.py
ghost_action_method = 'random' # 'random' or 'catchup', ghost using 'catchup' is more intelligent than 'random'.