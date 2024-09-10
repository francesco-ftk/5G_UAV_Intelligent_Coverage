import time
import gymnasium as gym
import numpy
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import math
import random

from gym_cruising.memory.replay_memory import ReplayMemory, Transition
from gym_cruising.neural_network.LSTM import LSTM
from gym_cruising.neural_network.deep_Q_net import DeepQNet
from gym_cruising.neural_network.transformer_encoder_decoder import TransformerEncoderDecoder

UAV_NUMBER = 3
GU_NUMBER = 60

TRAIN = False
EPS_START = 0.9  # the starting value of epsilon
EPS_END = 0.3  # the final value of epsilon
EPS_DECAY = 60000  # controls the rate of exponential decay of epsilon, higher means a slower decay
BATCH_SIZE = 256  # is the number of transitions random sampled from the replay buffer
LEARNING_RATE = 1e-4  # is the learning rate of the Adam optimizer, should decrease (1e-5)
BETA = 0.005  # is the update rate of the target network
GAMMA = 0.99  # Discount Factor

MAX_POSITION = 4000.0
MIN_POSITION = 0.0
MAX_SPEED_UAV = 5.86  # m/s

time_steps_done = -1
input_size_lstm = 16
hidden_size_lstm = 8
output_size_lstm = 2  # [μx, μy]
seq_len_lstm = 1
num_layer_lstm = 1
batch_size = 1

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("DEVICE:", device)

if TRAIN:
    env = gym.make('gym_cruising:Cruising-v0', render_mode='rgb_array', track_id=2)

    env.action_space.seed(42)
    state, info = env.reset(seed=int(time.perf_counter()))  # 42

    # ACTOR POLICY NET policy
    transformer_policy = TransformerEncoderDecoder().to(device)
    lstm_policy = LSTM(input_size_lstm, hidden_size_lstm, output_size_lstm, seq_len_lstm, num_layer_lstm).to(device)

    # CRITIC Q NET policy
    deep_Q_net_policy = DeepQNet(hidden_size_lstm + hidden_size_lstm, output_size_lstm).to(device)

    # COMMENT FOR INITIAL TRAINING
    # PATH_TRANSFORMER = '../neural_network/BaseTransformer.pth'
    # transformer_policy.load_state_dict(torch.load(PATH_TRANSFORMER))
    # PATH_LSTM = '../neural_network/BaseLSTM.pth'
    # lstm_policy.load_state_dict(torch.load(PATH_LSTM))
    # PATH_DEEP_Q = '../neural_network/BaseDeepQ.pth'
    # deep_Q_net_policy.load_state_dict(torch.load(PATH_DEEP_Q))

    # ACTOR POLICY NET target
    transformer_target = TransformerEncoderDecoder().to(device)  # TODO domanda 1
    lstm_target = LSTM(input_size_lstm, hidden_size_lstm, output_size_lstm, seq_len_lstm, num_layer_lstm).to(device)

    # CRITIC Q NET target
    deep_Q_net_target = DeepQNet(hidden_size_lstm + hidden_size_lstm, output_size_lstm).to(device)

    # set target parameters equal to main parameters
    transformer_target.load_state_dict(transformer_policy.state_dict())
    lstm_target.load_state_dict(lstm_policy.state_dict())
    deep_Q_net_target.load_state_dict(deep_Q_net_policy.state_dict())

    lstm_hidden_states_policy = torch.empty(UAV_NUMBER, batch_size, hidden_size_lstm)
    lstm_cell_states_policy = torch.empty(UAV_NUMBER, batch_size, hidden_size_lstm)
    for i in range(UAV_NUMBER):
        lstm_hidden_states_policy[i] = torch.zeros(num_layer_lstm, batch_size, hidden_size_lstm)
        lstm_cell_states_policy[i] = cell_state = torch.zeros(num_layer_lstm, batch_size, hidden_size_lstm)

    lstm_hidden_states_target = lstm_hidden_states_policy
    lstm_cell_states_target = lstm_cell_states_policy

    optimizer_transformer = optim.Adam(transformer_policy.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    optimizer_lstm = optim.Adam(lstm_policy.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    optimizer_deep_Q = optim.Adam(deep_Q_net_policy.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    replay_buffer = ReplayMemory(60000)


    def normalize(state: np.ndarray) -> np.ndarray:
        nornmalized_state = np.ndarray(shape=state.shape, dtype=np.float64)
        for i in range(len(state)):
            nornmalized_state[i] = (state[i] - MIN_POSITION) / (MAX_POSITION - MIN_POSITION)
        return nornmalized_state


    def select_actions_epsilon(tokens):
        action = []
        global time_steps_done
        global UAV_NUMBER
        time_steps_done += 1
        for i in range(UAV_NUMBER):
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * time_steps_done / EPS_DECAY)
            if sample > eps_threshold:
                # return action according to LSTM [μx, μy]
                output, (hs, cs) = lstm_policy(tokens[i], 1, lstm_hidden_states_policy[i].unsqueeze(0),
                                               lstm_cell_states_policy[i].unsqueeze(0))
                lstm_hidden_states_policy[i] = hs
                lstm_cell_states_policy[i] = cs
                output = output.cpu().numpy().reshape(2)
                output = output * MAX_SPEED_UAV + np.random.normal(0, 1)   # TODO domanda 2
                action.append(output)
            else:
                output = env.action_space.sample()[0]
                action.append(output)
        return action


    def optimize_model():

        global UAV_NUMBER

        if len(replay_buffer) < BATCH_SIZE:
            return
        transitions = replay_buffer.sample(BATCH_SIZE)

        # This converts batch-arrays of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        next_states_batch = torch.cat(batch.next_states)
        states_batch = torch.cat(batch.states)
        actions_batch = torch.cat(batch.actions)
        rewards_batch = torch.cat(batch.rewards)

        for i in range(UAV_NUMBER):  # TODO domanda 5, 6, 7
            # prendo nuovo stato, genero tokens con target transformer, genero azione con target lstm e uso questi per calcolare y
            target_q_values = rewards_batch[:, i].unsqueeze(1) + GAMMA * deep_Q_net_target(next_states_batch[:, i], actions_batch[:, i])  # TODO
            q_values = deep_Q_net_policy(states_batch[:, i], actions_batch[:, i])
            criterion = nn.SmoothL1Loss()  # nn.MSELoss()
            critic_loss = criterion(q_values, target_q_values)
            # Optimize the model
            optimizer_deep_Q.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_value_(deep_Q_net_policy.parameters(), 100)
            optimizer_deep_Q.step()

            actor_loss = -deep_Q_net_policy(states_batch[:, i], actions_batch[:, i]).mean()  # negativo per massimizzare Q
            optimizer_transformer.zero_grad()
            optimizer_lstm.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_value_(transformer_policy.parameters(), 100)
            torch.nn.utils.clip_grad_value_(lstm_policy.parameters(), 100)
            optimizer_transformer.step()
            optimizer_lstm.step()


    if torch.cuda.is_available():
        num_episodes = 2001
    else:
        num_episodes = 101

    # Writer will output to ./runs/ directory by default
    # writer = SummaryWriter("runs")

    print("START UAV COOPERATIVE COVERAGE TRAINING")

    for i_episode in range(0, num_episodes, 1):
        print("Episode: ", i_episode)
        state, info = env.reset(seed=int(time.perf_counter()))
        state = normalize(state)  # Normalize in [0,1]  # TODO domanda 3
        steps = 1
        while True:
            uav_positions, connected_gu_positions = np.split(state, [UAV_NUMBER], axis=0)
            uav_positions = torch.from_numpy(uav_positions).float().to(device)
            connected_gu_positions = torch.from_numpy(connected_gu_positions).float().to(device)
            tokens = transformer_policy(connected_gu_positions, uav_positions)

            actions = select_actions_epsilon(tokens)

            next_state, reward, terminated, truncated, _ = env.step(actions)

            if steps > 300:
                truncated = True
            done = terminated or truncated

            if not terminated and not truncated:  # TODO rimuovo esempi dove agenti escono dall'environment
                # Store the transition in memory
                next_state = normalize(next_state)
                replay_buffer.push(state, actions, next_state,
                                   reward)  # TODO domanda 4, che stato salvare, tokens, osservazione, cell e hidden states?
                # Move to the next state
                state = next_state
                # Perform one step of the optimization
                optimize_model()
                steps += 1

            if steps % 100 == 0:
                # Soft update of the target network's weights
                # Q′ ← β * Q + (1 − β) * Q′
                target_net_state_dict = transformer_target.state_dict()
                policy_net_state_dict = transformer_policy.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * BETA + target_net_state_dict[key] * (
                                1 - BETA)
                transformer_target.load_state_dict(target_net_state_dict)

                target_net_state_dict = lstm_target.state_dict()
                policy_net_state_dict = lstm_policy.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * BETA + target_net_state_dict[key] * (
                                1 - BETA)
                lstm_target.load_state_dict(target_net_state_dict)

                target_net_state_dict = deep_Q_net_target.state_dict()
                policy_net_state_dict = deep_Q_net_policy.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * BETA + target_net_state_dict[key] * (
                                1 - BETA)
                deep_Q_net_target.load_state_dict(target_net_state_dict)

            if done:
                break

        # if i_episode % 50 == 0:
        #     state, info = env.reset()
        #     steps = 0
        #     while True:
        #         state = normalize(state)
        #         state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        #         action = policy_net(state).max(1)[1].view(1, 1)
        #         state, reward, terminated, truncated, _ = env.step(action.item())
        #         steps += 1
        #         if steps >= 100:
        #             truncated = True
        #
        #         if terminated or truncated:
        #             # tensorboard --logdir=runs
        #             writer.add_scalars('Reward', {'policy_net': reward}, i_episode)
        #             break

    # save the policy nets
    torch.save(transformer_policy.state_dict(), '../neural_network/last_transformer.pth')
    torch.save(lstm_policy.state_dict(), '../neural_network/last_lstm.pth')
    torch.save(deep_Q_net_policy.state_dict(), '../neural_network/last_deep_q_net.pth')

    # writer.close()
    env.close()
    print('TRAINING COMPLETE')

else:

    env = gym.make('gym_cruising:Cruising-v0', render_mode='human', track_id=2)

    env.action_space.seed(42)
    state, info = env.reset(seed=int(time.perf_counter()))  # 42

    for _ in range(100):
        state, reward, terminated, truncated, info = env.step(0)
        print(f'observation={state} info={info}')

        if terminated:
            observation, info = env.reset()

    env.close()

    # policy_net = DQLN(n_observations, n_actions).to(device)
    # PATH = '../neural_network/Best.pth'
    # policy_net.load_state_dict(torch.load(PATH))
    # not_terminated = 0
    # success = 0
    # TEST_EPISODES = 100
