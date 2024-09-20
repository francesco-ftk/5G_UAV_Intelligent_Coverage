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
from gym_cruising.neural_network.MLP_policy_net import MLPPolicyNet
from gym_cruising.neural_network.deep_Q_net import DeepQNet
from gym_cruising.neural_network.transformer_encoder_decoder import TransformerEncoderDecoder

UAV_NUMBER = 1

TRAIN = True
EPS_START = 0.9  # the starting value of epsilon
EPS_END = 0.3  # the final value of epsilon
EPS_DECAY = 60000  # controls the rate of exponential decay of epsilon, higher means a slower decay
BATCH_SIZE = 256  # is the number of transitions random sampled from the replay buffer
LEARNING_RATE = 1e-4  # is the learning rate of the Adam optimizer, should decrease (1e-5)
BETA = 0.005  # is the update rate of the target network
GAMMA = 0.99  # Discount Factor

MAX_SPEED_UAV = 100.0  # 5.86  # m/s

time_steps_done = -1

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# print("DEVICE:", device)

if TRAIN:
    env = gym.make('gym_cruising:Cruising-v0', render_mode='rgb_array', track_id=2)
    env.action_space.seed(42)

    # ACTOR POLICY NET policy
    transformer_policy = TransformerEncoderDecoder().to(device)
    mlp_policy = MLPPolicyNet().to(device)

    # CRITIC Q NET policy
    deep_Q_net_policy = DeepQNet().to(device)

    # COMMENT FOR INITIAL TRAINING
    PATH_TRANSFORMER = './neural_network/lastTransformer.pth'
    transformer_policy.load_state_dict(torch.load(PATH_TRANSFORMER))
    PATH_MLP_POLICY = './neural_network/lastMLP.pth'
    mlp_policy.load_state_dict(torch.load(PATH_MLP_POLICY))
    PATH_DEEP_Q = './neural_network/lastDeepQ.pth'
    deep_Q_net_policy.load_state_dict(torch.load(PATH_DEEP_Q))

    # ACTOR POLICY NET target
    transformer_target = TransformerEncoderDecoder().to(device)
    mlp_target = MLPPolicyNet().to(device)

    # CRITIC Q NET target
    deep_Q_net_target = DeepQNet().to(device)

    # set target parameters equal to main parameters
    transformer_target.load_state_dict(transformer_policy.state_dict())
    mlp_target.load_state_dict(mlp_policy.state_dict())
    deep_Q_net_target.load_state_dict(deep_Q_net_policy.state_dict())

    optimizer_transformer = optim.Adam(transformer_policy.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    optimizer_mlp = optim.Adam(mlp_policy.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    optimizer_deep_Q = optim.Adam(deep_Q_net_policy.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    replay_buffer = ReplayMemory(10000)


    def select_actions_epsilon(state):
        global time_steps_done
        global UAV_NUMBER
        uav_info, connected_gu_positions = np.split(state, [UAV_NUMBER * 2], axis=0)
        uav_info = uav_info.reshape(UAV_NUMBER, 4)
        uav_info = torch.from_numpy(uav_info).float().to(device)
        connected_gu_positions = torch.from_numpy(connected_gu_positions).float().to(device)
        action = []
        with torch.no_grad():
            tokens = transformer_policy(connected_gu_positions, uav_info)
        time_steps_done += 1
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * time_steps_done / EPS_DECAY)
        for i in range(UAV_NUMBER):
            sample = random.random()
            if sample > eps_threshold:
                with torch.no_grad():
                    # return action according to MLP [vx, vy]
                    output = mlp_policy(tokens[i])
                    output = F.tanh(output + torch.randn(2).to(
                        device))  # TODO tanh messo fuori per fare prima aggiunta randomica, non serve anche in test?
                    output = output.cpu().numpy().reshape(2)
                    output = output * MAX_SPEED_UAV
                    action.append(output)
            else:
                output = env.action_space.sample()[0]  # TODO rivedere
                action.append(output)
        return action


    def optimize_model():

        torch.autograd.set_detect_anomaly(True)  # TODO detection

        global UAV_NUMBER
        global BATCH_SIZE

        if len(replay_buffer) < 3000:
            return
        transitions = replay_buffer.sample(BATCH_SIZE)

        # This converts batch-arrays of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        states_batch = batch.states
        actions_batch = batch.actions
        actions_batch = tuple(
            [torch.tensor(array, dtype=torch.float32) for array in sublist] for sublist in actions_batch)
        rewards_batch = batch.rewards
        rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).unsqueeze(1).to(device)
        next_states_batch = batch.next_states

        # prepare the batch of states
        state_uav_info_batch = tuple(np.split(array, [UAV_NUMBER * 2], axis=0)[0] for array in states_batch)
        state_connected_gu_positions_batch = tuple(
            np.split(array, [UAV_NUMBER * 2], axis=0)[1] for array in states_batch)
        state_uav_info_batch = tuple(array.reshape(UAV_NUMBER, 4) for array in state_uav_info_batch)
        state_uav_info_batch = tuple(torch.from_numpy(array).float().to(device) for array in state_uav_info_batch)
        state_connected_gu_positions_batch = tuple(
            torch.from_numpy(array).float().to(device) for array in state_connected_gu_positions_batch)

        # prepare the batch of next states
        next_state_uav_info_batch = tuple(
            np.split(array, [UAV_NUMBER * 2], axis=0)[0] for array in next_states_batch)
        next_state_connected_gu_positions_batch = tuple(
            np.split(array, [UAV_NUMBER * 2], axis=0)[1] for array in next_states_batch)
        next_state_uav_info_batch = tuple(array.reshape(UAV_NUMBER, 4) for array in next_state_uav_info_batch)
        next_state_uav_info_batch = tuple(
            torch.from_numpy(array).float().to(device) for array in next_state_uav_info_batch)
        next_state_connected_gu_positions_batch = tuple(
            torch.from_numpy(array).float().to(device) for array in next_state_connected_gu_positions_batch)

        # get tokens from batch of states and next states
        tokens_batch_next_states_target = []  # tokens batch from next states batch with target transformer [BATCH_SIZE, UAV_NUMBER, 16]
        tokens_batch_states = []  # tokens batch from states batch with policy transformer [BATCH_SIZE, UAV_NUMBER, 16]
        tokens_batch_states_target = []
        for j in range(BATCH_SIZE):
            with torch.no_grad():
                tokens_batch_next_states_target.append(
                    transformer_target(next_state_connected_gu_positions_batch[j], next_state_uav_info_batch[j]))
                tokens_batch_states_target.append(
                    transformer_target(state_connected_gu_positions_batch[j], state_uav_info_batch[j]))
            tokens_batch_states.append(
                transformer_policy(state_connected_gu_positions_batch[j], state_uav_info_batch[j]))

        for i in range(UAV_NUMBER):
            # UPDATE Q-FUNCTION
            with torch.no_grad():
                # Concatenate i-th UAV's tokens along the batch size [BATCH_SIZE, 1, 16]
                current_batch_tensor_tokens_next_states_target = torch.cat(
                    [tensor[0].unsqueeze(0) for tensor in tokens_batch_next_states_target],
                    dim=0)
                output_batch = mlp_target(current_batch_tensor_tokens_next_states_target)
                output_batch = F.tanh(output_batch + torch.randn(BATCH_SIZE, 2).to(device))
                output_batch = output_batch * MAX_SPEED_UAV  # actions batch for UAV i-th [BATCH_SIZE, 2]
                current_y_batch = rewards_batch + GAMMA * deep_Q_net_target(
                    current_batch_tensor_tokens_next_states_target, output_batch)
            # Concatenate i-th UAV's tokens along the batch size [BATCH_SIZE, 1, 16]
            current_batch_tensor_tokens_states = torch.cat(
                [tensor[0].unsqueeze(0) for tensor in tokens_batch_states],
                dim=0)
            # Concatenate i-th UAV's actions along the batch size [BATCH_SIZE, 2]
            current_batch_actions = torch.cat(
                [action[0].unsqueeze(0) for action in actions_batch],
                dim=0).to(device)
            Q_values_batch = deep_Q_net_policy(current_batch_tensor_tokens_states, current_batch_actions)

            criterion = nn.MSELoss()
            # Optimize Deep Q Net
            loss_Q = criterion(Q_values_batch, current_y_batch)
            optimizer_deep_Q.zero_grad()
            loss_Q.backward(retain_graph=True)  # TODO messi senno errore
            torch.nn.utils.clip_grad_value_(deep_Q_net_policy.parameters(), 100)
            optimizer_deep_Q.step()
            # Optimize Transformer Net
            loss_transformer = criterion(Q_values_batch, current_y_batch)
            optimizer_transformer.zero_grad()
            loss_transformer.backward(retain_graph=True)  # TODO messi senno errore
            torch.nn.utils.clip_grad_value_(transformer_policy.parameters(), 100)
            optimizer_transformer.step()

            # UPDATE POLICY
            # Concatenate i-th UAV's tokens along the batch size [BATCH_SIZE, 1, 16]
            current_batch_tensor_tokens_states_target = torch.cat(
                [tensor[0].unsqueeze(0) for tensor in tokens_batch_states_target],
                dim=0)
            output_batch = mlp_policy(current_batch_tensor_tokens_states_target)
            output_batch = F.tanh(output_batch + torch.randn(BATCH_SIZE, 2).to(device))
            output_batch = output_batch * MAX_SPEED_UAV  # actions batch for UAV i-th [BATCH_SIZE, 2]
            Q_values_batch = deep_Q_net_policy(current_batch_tensor_tokens_states_target, output_batch)
            loss_Policy = -Q_values_batch.mean()

            # Optimize Policy Net MLP
            optimizer_mlp.zero_grad()
            loss_Policy.backward(retain_graph=True)  # TODO messi senno errore
            torch.nn.utils.clip_grad_value_(mlp_policy.parameters(), 100)
            optimizer_mlp.step()


    if torch.cuda.is_available():
        num_episodes = 100
    else:
        num_episodes = 100

    # Writer will output to ./runs/ directory by default
    # writer = SummaryWriter("runs")

    print("START UAV COOPERATIVE COVERAGE TRAINING")

    for i_episode in range(0, num_episodes, 1):
        print("Episode: ", i_episode)
        state, info = env.reset(seed=int(time.perf_counter()))
        steps = 1
        while True:
            actions = select_actions_epsilon(state)
            next_state, reward, terminated, truncated, _ = env.step(actions)

            if steps == 300:
                truncated = True
            done = terminated or truncated

            if not terminated:
                # Store the transition in memory
                replay_buffer.push(state, actions, next_state, reward)
                # Move to the next state
                state = next_state
                # Perform one step of the optimization
                optimize_model()
                steps += 1

            if len(replay_buffer) >= 3000 and steps % 50 == 0:
                # Soft update of the target network's weights
                # Q′ ← β * Q + (1 − β) * Q′
                target_net_state_dict = transformer_target.state_dict()
                policy_net_state_dict = transformer_policy.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * BETA + target_net_state_dict[key] * (
                            1 - BETA)
                transformer_target.load_state_dict(target_net_state_dict)

                target_net_state_dict = mlp_target.state_dict()
                policy_net_state_dict = mlp_policy.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * BETA + target_net_state_dict[key] * (
                            1 - BETA)
                mlp_target.load_state_dict(target_net_state_dict)

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
    torch.save(transformer_policy.state_dict(), './neural_network/lastTransformer.pth')
    torch.save(mlp_policy.state_dict(), './neural_network/lastMLP.pth')
    torch.save(deep_Q_net_policy.state_dict(), './neural_network/lastDeepQ.pth')

    # writer.close()
    env.close()
    print('TRAINING COMPLETE')

else:

    def select_actions(state):
        global UAV_NUMBER
        uav_info, connected_gu_positions = np.split(state, [UAV_NUMBER * 2], axis=0)
        uav_info = uav_info.reshape(UAV_NUMBER, 4)
        uav_info = torch.from_numpy(uav_info).float().to(device)
        connected_gu_positions = torch.from_numpy(connected_gu_positions).float().to(device)
        action = []
        with torch.no_grad():
            tokens = transformer_policy(connected_gu_positions, uav_info)
        for i in range(UAV_NUMBER):
            with torch.no_grad():
                # return action according to MLP [vx, vy]
                output = mlp_policy(tokens[i])
                output = F.tanh(output)  # TODO qui non aggiungo errore da distribuzione Normale?
                output = output.cpu().numpy().reshape(2)
                output = output * MAX_SPEED_UAV
                action.append(output)
        return action


    # For accuracy check
    # env = env = gym.make('gym_cruising:Cruising-v0', render_mode='rgb_array', track_id=2)
    # For visible check
    env = env = gym.make('gym_cruising:Cruising-v0', render_mode='human', track_id=2)

    env.action_space.seed(42)

    # ACTOR POLICY NET policy
    transformer_policy = TransformerEncoderDecoder().to(device)
    mlp_policy = MLPPolicyNet().to(device)

    PATH_TRANSFORMER = './neural_network/lastTransformer.pth'
    transformer_policy.load_state_dict(torch.load(PATH_TRANSFORMER))
    PATH_MLP_POLICY = './neural_network/lastMLP.pth'
    mlp_policy.load_state_dict(torch.load(PATH_MLP_POLICY))

    terminated = 0

    TEST_EPISODES = 100
    for j in range(TEST_EPISODES):
        print("Episode: ", j)
        state, info = env.reset(seed=int(time.perf_counter()))
        steps = 1
        max_reward = 0.0
        while True:
            actions = select_actions(state)
            next_state, reward, terminated, truncated, _ = env.step(actions)
            if reward > max_reward:
                max_reward = reward

            if steps == 300:
                truncated = True
            done = terminated or truncated

            state = next_state
            steps += 1

            if done:
                if terminated:
                    terminated += 1
                    print("Episode " + str(j) + " TERMINATED with max reward: " + str(max_reward))
                else:
                    print("Max reward in episode " + str(j) + ": " + str(max_reward))
                break

    env.close()
    print("Executed " + str(TEST_EPISODES) + " episodes:\n" + str(terminated) + " terminated\n" + str(
        TEST_EPISODES - terminated) + " episodes completed\n")
