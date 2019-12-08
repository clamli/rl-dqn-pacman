import os
import random
import torch
import torch.nn as nn
import config
from gameAPI.game import GamePacmanAgent
from ReplayMemory import ReplayMemory, PrioritizedReplayMemory

from utils import convert_idx_to_2_dim_tensor, convert_2_dim_tensor_to_4_dim_tensor, frames_to_tensor, single_frame_to_tensor, save_data

# torch.manual_seed(9001)
# random.seed(9001)


FloatTensor = torch.cuda.FloatTensor if config.use_cuda else torch.FloatTensor

class DoubleDQNAgent:

    def __init__(self, net: torch.nn.Module):
        self.game_agent = GamePacmanAgent(config)
        self.net = net
        if config.use_cuda:
            self.net = self.net.cuda()

    def train(self, foldname):
        if not os.path.exists(foldname):
            os.mkdir(foldname)
        copy_net = type(self.net)()
        if config.use_cuda:
            copy_net = copy_net.cuda()
        copy_net.load_state_dict(self.net.state_dict())
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
        loss_lst = []
        score_lst = []
        game_round = 1
        live_time = 0

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

                (position_lst, state_lst, action_lst, reward_lst, is_gameover_lst,
                 next_state_lst) = replay_memory.sample(
                    config.sample_size)

                prob = max(config.eps_start - (config.eps_start - config.eps_end) / config.eps_num_steps * count,
                           config.eps_end)
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
            if is_gameover or live_time >= config.timeout:
                self.game_agent.reset()
                game_round += 1
                live_time = 0
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
                    v_s = torch.max(q_t, dim=1)[0]
                    v_s_p = torch.max(q_t_p, dim=1)[0]
                    td_errors = abs(config.gamma * v_s_p + FloatTensor(reward_lst) - v_s)
                    for position, error in zip(position_lst, td_errors):
                        replay_memory.update(position, error)
                q_t_p = torch.max(q_t_p, dim=1)[1]
                new_q_t_p = copy_net(frames_to_tensor(next_state_lst))
                new_q_t_p = new_q_t_p.gather(1, q_t_p.view(-1, 1))
                new_q_t_p = new_q_t_p.view(-1)
                loss = mse_loss(
                    FloatTensor(reward_lst) + (1 - FloatTensor(is_gameover_lst)) * (config.gamma * new_q_t_p),
                    (q_t * FloatTensor(action_lst)).sum(1)
                )
                loss.backward()
                loss_lst.append(torch.Tensor.item(loss))

                if count % config.ddqn_replace == 0:
                    copy_net.load_state_dict(self.net.state_dict())

                prob = max(config.eps_start - (
                        config.eps_start - config.eps_end) / config.eps_num_steps * count,
                           config.eps_end)
                print(
                    "game round: %d, live_time: %d, iteration: %d, loss: %.4f, action: %s, current game score: %d, prob %f" % (
                    game_round, live_time, count, loss, str(action), self.game_agent.score, prob))

                optimizer.step()
                if count % config.save_model_threshold == 0:
                    torch.save(self.net.state_dict(), './' + foldname + '/model' + str(count) + '.pkl')
                    save_data({"loss_lst": loss_lst, "score_lst": score_lst}, './' + foldname + "/result" + str(count))

            frame = next_frame


    def test(self):
        count = 0
        score_lst = []
        game_round = 1
        live_time = 0

        # init
        frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(
            action=None)
        while is_gameover:
            self.game_agent.reset()
            frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(
                action=None)
        time_rem = config.timeout
        while True:
            count += 1
            time_rem -= 1

            prob = 0.1
            random_value = random.random()
            if random_value <= prob:
                action = None
            else:
                with torch.no_grad():
                    actions_values = self.net(
                        single_frame_to_tensor(frame))
                    max_value, action = torch.max(actions_values,
                                                  dim=1)
                    action = convert_idx_to_2_dim_tensor(action[0])

            next_frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(
                action=action)
            score_lst.append(self.game_agent.score)
            if is_gameover or time_rem <= 0:
                self.game_agent.reset()
                time_rem = config.timeout

                game_round += 1
                live_time = 0

            else:
                live_time += 1

            prob = max(config.eps_start - (
                    config.eps_start - config.eps_end) / config.eps_num_steps * count,
                       config.eps_end)
            print(
                "game round: %d, live_time: %d, iteration: %d, loss: %.4f, action: %s, current game score: %d, prob %f" % (
                    game_round, live_time, count, 0,
                    str(action), self.game_agent.score, prob))

            frame = next_frame
