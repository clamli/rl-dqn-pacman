import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import config
from gameAPI.game import GamePacmanAgent
from collections import deque

from sum_tree import SumTree
from utils import convert_idx_to_2_dim_tensor, convert_2_dim_tensor_to_4_dim_tensor, frames_to_tensor, single_frame_to_tensor, save_data

# torch.manual_seed(9001)
# random.seed(9001)


FloatTensor = torch.cuda.FloatTensor if config.use_cuda else torch.FloatTensor


class ReplayMemory:

    def __init__(self):
        self.memory = []

    def add(self, state, action, reward, is_gameover, next_state):
        if len(self.memory) >= config.max_memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, is_gameover, next_state))

    def sample(self, batch_size):
        rand_ind = random.sample(range(0, len(self.memory)), min(batch_size, len(self.memory)))
        state_lst = [self.memory[i][0] for i in rand_ind]
        action_lst = [self.memory[i][1] for i in rand_ind]
        reward_lst = [self.memory[i][2] for i in rand_ind]
        is_gameover_lst = [self.memory[i][3] for i in rand_ind]
        next_state_lst = [self.memory[i][4] for i in rand_ind]
        return [], state_lst, action_lst, reward_lst, is_gameover_lst, next_state_lst


class PrioritizedReplayMemory:
    e = 0.01
    a = 0.6
    abs_err_upper = 1.

    def __init__(self, memory_size):
        self.memory = SumTree(memory_size)

    def add(self, state, action, reward, is_gameover, next_state):
        p = np.max(self.memory.tree[-self.memory.memory_size])
        if p == 0:
            p = self.abs_err_upper
        transition = (state, action, reward, is_gameover, next_state)
        self.memory.add(p, transition)

    def update(self, position, error):
        p = (error + self.e) ** self.a
        self.memory.update(position, p)

    def sample(self, sample_size):
        state_lst = []
        action_lst = []
        reward_lst = []
        is_gameover_lst = []
        next_state_lst = []
        positions_lst = []
        segment = self.memory.total() / sample_size
        for i in range(sample_size):
            a = segment * i
            b = segment * (i+1)
            s = random.uniform(a, b)
            position, _, data = self.memory.get(s)
            positions_lst.append(position)
            state_lst.append(data[0])
            action_lst.append(data[1])
            reward_lst.append(data[2])
            is_gameover_lst.append(data[3])
            next_state_lst.append(data[4])
        return positions_lst, state_lst, action_lst, reward_lst, is_gameover_lst, next_state_lst


class DQN(torch.nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        #self.model = torchvision.models.alexnet(pretrained=True)
        #self.model.classifier[6] = nn.Linear(4096, num_actions)

        self.resnet18 = torchvision.models.resnet18()
        in_ch = 3
        if config.use_simple:
            in_ch = 1
        self.resnet18.conv1 = nn.Conv2d(
            in_channels=in_ch, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=4)

    def forward(self, x):
        #x = self.model(x)
        x = self.resnet18(x)
        return x


class Agent:

    def __init__(self, net: torch.nn.Module):
        self.game_agent = GamePacmanAgent(config)
        self.net = net
        if config.use_cuda:
            self.net = self.net.cuda()

    # def train2(self):
    #     action_pred = None
    #     image = None
    #     frames = []
    #     game_memories = deque()
    #     optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
    #
    #     mse_loss = nn.MSELoss(reduction='elementwise_mean')
    #     num_iter = 0
    #     num_games = 0
    #     num_wins = 0
    #     num_games_lst = []
    #     num_wins_lst = []
    #     score_lst = []
    #     loss_lst = []
    #     while True:
    #         if len(game_memories) > config.max_memory_size:
    #             game_memories.popleft()
    #         frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(action=action_pred)
    #         score_lst.append(self.game_agent.score)
    #
    #         if is_gameover:
    #             self.game_agent.reset()
    #             if len(game_memories) >= config.start_training_threshold:
    #                 num_games += 1
    #                 num_wins += int(is_win)
    #                 num_games_lst.append(num_games)
    #                 num_wins_lst.append(num_wins)
    #         frames.append(frame)
    #         if len(frames) == 1:
    #             image_prev = image
    #             image = np.concatenate(frames, -1)
    #             frames.pop(0)
    #             if image_prev is not None:
    #                 game_memories.append((image, image_prev, reward, convert_2_dim_tensor_to_4_dim_tensor(action),
    #                                   is_gameover))
    #
    #         # explore
    #         if len(game_memories) < config.start_training_threshold:
    #             print('[STATE]: explore, [MEMORYLEN]: %d' % len(game_memories))
    #
    #         # train
    #         else:
    #             # --get data
    #             num_iter += 1
    #             images_input = []
    #             images_prev_input = []
    #             reward_lst = []
    #             action_lst = []
    #             is_gameover_lst = []
    #             for each in random.sample(game_memories, config.sample_size):
    #                 image_input = each[0].astype(np.float32) / 255.
    #                 image_input.resize((1, *image_input.shape))
    #                 images_input.append(image_input)
    #                 image_prev_input = each[1].astype(np.float32) / 255.
    #                 image_prev_input.resize((1, *image_prev_input.shape))
    #                 images_prev_input.append(image_prev_input)
    #                 reward_lst.append(each[2])
    #                 action_lst.append(each[3])
    #                 is_gameover_lst.append(each[4])
    #
    #             images_input_torch = torch.from_numpy(np.concatenate(images_input, 0)).permute(0, 3, 1, 2).type(
    #                 FloatTensor)
    #             images_prev_input_torch = torch.from_numpy(np.concatenate(images_prev_input, 0)).permute(0, 3, 1,
    #                                                                                                      2).type(
    #                 FloatTensor)
    #
    #             # --compute loss
    #             optimizer.zero_grad()
    #             q_t = self.net(images_input_torch)
    #             q_t = torch.max(q_t, dim=1)[0]
    #             loss = mse_loss(
    #                 torch.Tensor(reward_lst).type(FloatTensor) + (1 - torch.Tensor(is_gameover_lst).type(FloatTensor)) * (
    #                             0.95 * q_t),
    #                 (self.net(images_prev_input_torch) * torch.Tensor(action_lst).type(FloatTensor)).sum(1))
    #             loss.backward()
    #             loss_lst.append(torch.Tensor.item(loss))
    #             optimizer.step()
    #             # --make decision
    #             prob = max(config.eps_start - (
    #                         config.eps_start - config.eps_end) / config.eps_num_steps * num_iter,
    #                        config.eps_end)
    #             rand_value = random.random()
    #             if rand_value > prob:
    #                 with torch.no_grad():
    #                     image_input = image.astype(np.float32) / 255.
    #                     image_input.resize((1, *image_input.shape))
    #                     image_input_torch = torch.from_numpy(image_input).permute(0, 3, 1, 2).type(FloatTensor)
    #                     action_pred = self.net(image_input_torch)
    #                     max_value, action = torch.max(action_pred, dim=1)
    #                     action_pred = convert_idx_to_2_dim_tensor(action[0])
    #                     #actions_values = self.net(single_frame_to_tensor(frame))
    #                     #max_value, action = torch.max(actions_values, dim=1)
    #                     #action_pred = convert_idx_to_2_dim_tensor(action[0])
    #             else:
    #                 action_pred = None
    #
    #             print('[STATE]: training, [ITER]: %d, [LOSS]: %.3f, [ACTION]: %s' % (num_iter, loss.item(), str(action_pred)))
    #             if num_iter % config.save_model_threshold == 0:
    #                 torch.save(self.net.state_dict(), str(num_iter) + '.pkl')
    #                 save_data({"num_games_lst": num_games_lst,
    #                            "num_wins_lst": num_wins_lst,
    #                            "loss_lst": loss_lst, "score_lst": score_lst},
    #                           "result" + str(num_iter))


    #def get_td_error(self, frame, reward, next_frame):
    #    v_s = torch.max(self.net(single_frame_to_tensor(frame))).item()
    #    v_s_p_1 = torch.max(self.net(single_frame_to_tensor(next_frame))).item()
    #    return abs(config.gamma * v_s_p_1 + reward - v_s)

    def train(self):
        if config.use_per:
            replay_memory = PrioritizedReplayMemory(config.max_memory_size)
        else:
            replay_memory = ReplayMemory()
        if config.use_simple:
            optimizer = torch.optim.Adam(self.net.parameters())
        else:
            optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        mse_loss = nn.MSELoss(reduction='elementwise_mean')
        count = 0
        num_games = 0
        num_wins = 0
        num_games_lst = []
        num_wins_lst = []
        loss_lst = []
        score_lst = []
        game_round = 1
        live_time = 0
        for episode in range(1, config.M+1):
            # init
            frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(action=None)
            while is_gameover:
                self.game_agent.reset()
                frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(action=None)

            while True:
                state_lst = []
                action_lst = []
                reward_lst = []
                is_gameover_lst = []
                next_state_lst = []
                position_lst = []
                is_train = len(replay_memory.memory) >= config.start_training_threshold
                if not is_train:
                    action = None
                else:
                    count += 1

                    (position_lst, state_lst, action_lst, reward_lst, is_gameover_lst, next_state_lst) = replay_memory.sample(
                        config.sample_size)

                    prob = max(config.eps_start - (config.eps_start - config.eps_end) / config.eps_num_steps * count, config.eps_end)
                    random_value = random.random()
                    if random_value <= prob:
                        action = None
                    else:
                        with torch.no_grad():
                            actions_values = self.net(single_frame_to_tensor(frame))
                            max_value, action = torch.max(actions_values, dim=1)
                            action = convert_idx_to_2_dim_tensor(action[0])

                next_frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(action=action)
                score_lst.append(self.game_agent.score)
                if is_gameover:
                    self.game_agent.reset()

                    game_round += 1
                    live_time = 0

                    if len(replay_memory.memory) >= config.start_training_threshold:
                        num_games += 1
                        num_wins += int(is_win)
                        num_games_lst.append(num_games)
                        num_wins_lst.append(num_wins)

                else:
                    live_time += 1

                replay_memory.add(frame, convert_2_dim_tensor_to_4_dim_tensor(action), reward, is_gameover, next_frame)

                if is_train:
                    # v_s = torch.max(self.net(single_frame_to_tensor(frame))).item()
                    # v_s_p_1 = torch.max(self.net(single_frame_to_tensor(next_frame))).item()
                    # return abs(config.gamma * v_s_p_1 + reward - v_s)
                    # sampling
                    optimizer.zero_grad()
                    q_t = self.net(frames_to_tensor(state_lst))
                    q_t_p = self.net(frames_to_tensor(next_state_lst))
                    if config.use_per:
                        v_s = torch.max(q_t)
                        v_s_p = torch.max(q_t_p)
                        td_errors = abs(config.gamma * v_s_p + reward - v_s)
                        for position, error in zip(position_lst, td_errors):
                            replay_memory.update(position, error)
                    q_t_p = torch.max(q_t_p, dim=1)[0]
                    loss = mse_loss(
                        FloatTensor(reward_lst) + (1 - FloatTensor(is_gameover_lst)) * (config.gamma * q_t_p),
                        (q_t * FloatTensor(action_lst)).sum(1)
                    )
                    loss.backward()
                    loss_lst.append(torch.Tensor.item(loss))

                    prob = max(config.eps_start - (
                                config.eps_start - config.eps_end) / config.eps_num_steps * count,
                               config.eps_end)
                    print("episode: %d, game round: %d, live_time: %d, iteration: %d, loss: %.4f, action: %s, current game score: %d, prob %f" % (episode, game_round, live_time, count, loss, str(action), self.game_agent.score, prob))

                    optimizer.step()
                    if count % config.save_model_threshold == 0:
                        torch.save(self.net.state_dict(), './models/model' + str(count) + '.pkl')
                        save_data({"num_games_lst": num_games_lst, "num_wins_lst": num_wins_lst, "loss_lst": loss_lst, "score_lst": score_lst}, "result" + str(count))

                frame = next_frame


    def test(self):
        action = None
        while True:
            frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(action=action)
            if is_gameover:
                self.game_agent.reset()
            if random.random() <= config.eps_end:
                action = None
            else:
                with torch.no_grad():
                    actions_values = self.net(single_frame_to_tensor(frame))
                    max_value, action = torch.max(actions_values, dim=1)
                    action = convert_idx_to_2_dim_tensor(action)
            print('[ACTION]: %s, [SCORE]: %d' % (str(action), self.game_agent.score))
