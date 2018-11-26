import gym
import numpy as np
import pickle
import os
print(os.getcwd())
# import mxnet
# import minpy
# import minpy.numpy as np
# from minpy.context import set_context, cpu, gpu
# set_context(gpu(0))


def preprocess_observations(input_observation, prev_processed_observation, dimensions):
    processed_observation = input_observation[35:195]
    processed_observation = processed_observation[::2, ::2, :]
    processed_observation = processed_observation[:, :, 0]
    processed_observation[processed_observation == 144] = 0
    processed_observation[processed_observation == 109] = 0
    processed_observation[processed_observation != 0] = 1 
    processed_observation = processed_observation.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations

def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
        weights[layer_name] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name]) # reset batch gradient buffer

def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
    return {
        '1': dC_dw1,
        '2': dC_dw2
    }

def apply_neural_nets(observation_matrix, weights):
    """ Based on the observation_matrix and weights, compute the new hidden layer values and the new output layer values"""
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values

def discount_rewards(rewards, gamma):
    """ Actions you took 20 steps before the end result are less important to the overall result than an action you took a step ago.
    This implements that logic by discounting the reward on previous actions based on how long ago they were taken"""
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

def discount_with_rewards(gradient_log_p, episode_rewards, gamma):
    """ discount the gradient with the normalized rewards """
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(vector):
    vector[vector < 0] = 0
    return vector

def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        # signifies up in openai gym
        return 2
    else:
         # signifies down in openai gym
        return 3

def main():
    enviroment = gym.make("Pong-v0")
    observation = enviroment.reset()
    rounds = 10
    decay = 0.99
    gamma = 0.99
    neurons = 200
    learning = 1e-4
    dimensions = 80*80

    episode = 0
    reward_sum = 0
    running_reward = None
    prev_observations = None

    #weights = {
        #"1": np.random.randn(neurons, dimensions) / np.sqrt(dimensions),
        #"2": np.random.randn(neurons) / np.sqrt(neurons)
    #}
    with open('pesos30.txt', 'rb') as pesos:
       weights = pickle.load(pesos)
       print("Carregou")
    print (weights)
    expectation_g_squared = {}
    g_dict = {}

    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []

    while True:
        #enviroment.render()
        processed_observations, prev_observations = preprocess_observations(observation, prev_observations, dimensions)
        hidden_layer_values, up_probability = apply_neural_nets(processed_observations, weights)
        episode_observations.append(processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)
        action = choose_action(up_probability)
        # carry out the chosen action
        observation, reward, done, info = enviroment.step(action)

        reward_sum += reward
        episode_rewards.append(reward)

        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability
        episode_gradient_log_ps.append(loss_function_gradient)

        if done:
            episode = episode + 1
            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_observations = np.vstack(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)

            # Tweak the gradient of the log_ps based on the discounted rewards
            episode_gradient_log_ps_discounted = discount_with_rewards(episode_gradient_log_ps, episode_rewards, gamma)
            gradient = compute_gradient(episode_gradient_log_ps_discounted, episode_hidden_layer_values, episode_observations, weights)

            for layer_name in gradient:
                g_dict[layer_name] += gradient[layer_name]

            if episode % rounds == 0:
                if episode == 100:
                    episode = 10
                update_weights(weights, expectation_g_squared, g_dict, decay, learning)
                pesos = "pesos" + str(episode) + ".txt"
                with open(pesos, 'wb') as handle:
                    pickle.dump(weights, handle)
                    print("Salvou")
                print(weights)

            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], [] # reset values
            observation = enviroment.reset() # reset env
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print ("resetting env. episode reward total was {} running mean: {}".format(reward_sum, running_reward))
            reward_sum = 0
            prev_observations = None

main()