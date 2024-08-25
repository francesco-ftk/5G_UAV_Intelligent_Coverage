import time
import gymnasium as gym
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
from gym_cruising.neural_network.custom_transformer_encoder_decoder import CustomTransformerEncoderDecoder

UAV_NUMBER = 3
GU_NUMBER = 60

TRAIN = True
EPS_START = 0.9  # the starting value of epsilon
EPS_END = 0.3  # the final value of epsilon
EPS_DECAY = 60000  # controls the rate of exponential decay of epsilon, higher means a slower decay
BATCH_SIZE = 128  # is the number of transitions random sampled from the replay buffer
LEARNING_RATE = 1e-4  # is the learning rate of the Adam optimizer, should decrease (1e-5)

MAX_POSITION = 4000.0
MIN_POSITION = 0.0

time_steps_done = 0
input_size_lstm = 16
hidden_size_lstm = 8
output_size_lstm = 2
seq_len_lstm = 1
num_layer_lstm = 1
batch_size = 117

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("DEVICE:", device)

if TRAIN:
    env = gym.make('gym_cruising:Cruising-v0', render_mode='rgb_array', track_id=2)

    env.action_space.seed(42)
    state, info = env.reset(seed=int(time.perf_counter()))  # 42

    transformer_encoder_decoder_net = CustomTransformerEncoderDecoder()
    lstm_net = LSTM(input_size_lstm, hidden_size_lstm, output_size_lstm, seq_len_lstm, num_layer_lstm)
    token_hidden_states = torch.empty(UAV_NUMBER, batch_size, hidden_size_lstm)
    for i in range(UAV_NUMBER):
        token_hidden_states[i] = torch.zeros(num_layer_lstm, batch_size, hidden_size_lstm)

    # COMMENT FOR INITIAL TRAINING
    # PATH = '../neural_network/Base.pth'
    # transformer_encoder_decoder_net.load_state_dict(torch.load(PATH))
    # lstm_net.load_state_dict(torch.load(PATH))

    # optimizer = optim.Adam(lstm_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    # replay_buffer = ReplayMemory(6000)


    def normalize(state: np.ndarray) -> np.ndarray:
        nornmalized_state = np.ndarray(shape=state.shape, dtype=np.float64)
        for i in range(len(state)):
            nornmalized_state[i] = (state[i] - MIN_POSITION) / (MAX_POSITION - MIN_POSITION)
        return nornmalized_state


    def select_actions_epsilon(state):
        action = []
        global time_steps_done
        global UAV_NUMBER
        time_steps_done += 1
        for i in range(UAV_NUMBER):
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * time_steps_done / EPS_DECAY)
            if sample > eps_threshold:
                with torch.no_grad():
                    # return mean and covariance according to LSTM
                    output, token_hidden_states[i] = lstm_net(state, 1, token_hidden_states[i])
                    action.append(output)  # FIXME
            else:
                torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)  # TODO
        return action


    # def optimize_model():
    #     if len(replay_buffer) < BATCH_SIZE:
    #         return
    #     transitions = replay_buffer.sample(BATCH_SIZE)
    #
    #     # This converts batch-arrays of Transitions to Transition of batch-arrays.
    #     batch = Transition(*zip(*transitions))
    #
    #     # map() function returns a map object of the results after applying the given function to each item of a given iterable
    #     non_final_states_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
    #                                          dtype=torch.bool)
    #     non_final_next_states = torch.cat(
    #         [s for s in batch.next_state if s is not None])
    #     state_batch = torch.cat(batch.state)
    #     action_batch = torch.cat(batch.action)
    #     reward_batch = torch.cat(batch.reward)
    #
    #     # policy_net computes Q(state, action taken)
    #     state_action_values = policy_net(state_batch).gather(1, action_batch)
    #
    #     # This is merged based on the mask, such that we'll have either the expected
    #     # state value or 0 in case the state was final.
    #     # target_net computes max over actions of Q(next_state, action) for all next states
    #     next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #     with torch.no_grad():
    #         next_state_values[non_final_states_mask] = target_net(non_final_next_states).max(1)[0]
    #     # Compute the expected Q values with BELLMAN OPTIMALITY Q VALUE EQUATION:
    #     # Q(state,action) = reward(state,action) + GAMMA * max(Q(next_state, actions), action)
    #     expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    #
    #     criterion = nn.SmoothL1Loss()
    #     loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    #
    #     # Optimize the model
    #     optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    #     optimizer.step()

    if torch.cuda.is_available():
        num_episodes = 11
    else:
        num_episodes = 10

    # Writer will output to ./runs/ directory by default
    # writer = SummaryWriter("runs")

    print("START UAV COOPERATIVE COVERAGE TRAINING")

    for i_episode in range(0, num_episodes, 1):
        print("Episode: ", i_episode)
        state, info = env.reset(seed=int(time.perf_counter()))
        state = normalize(state)
        uav_positions, connected_gu_positions = np.split(state, [UAV_NUMBER], axis=0)
        uav_positions = torch.from_numpy(uav_positions).float()
        connected_gu_positions = torch.from_numpy(connected_gu_positions).float()
        # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        steps = 0
        while True:
            tokens = transformer_encoder_decoder_net(connected_gu_positions, uav_positions)
            # TODO inserire transformer e LSTM
            actions = select_actions_epsilon(state)  # TODO capire come fatta azione
            next_state, reward, terminated, truncated, _ = env.step(actions)
            next_state = normalize(next_state)  # Normalize in [0,1]
            if steps >= 120:
                if not terminated:
                    truncated = True
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated or truncated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            # replay_buffer.push(state, actions, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization
            # optimize_model()
            steps += 1

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

    # PATH = '../neural_network/last.pth'
    # torch.save(lstm_net.state_dict(), PATH)
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
