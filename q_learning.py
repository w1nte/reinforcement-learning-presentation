from mdp import PresentationEnvironment, simulate, policy_iteration
import matplotlib.pyplot as plt
import numpy as np
import random


def q_learning(env, episodes, steps_per_episode, alpha, epsilon, gamma, epsilon_decay=-0.0001):
    q_table = np.zeros((env.number_of_states, env.number_of_actions))
    rewards = []

    for episode in range(episodes):
        env.reset()

        total_rewards = 0
        state = env.current_state

        for step in range(steps_per_episode):
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.action_space)
            else:
                action = np.argmax(q_table[state, :])
                if action not in env.action_space:  # skip step if selected action is invalid
                    continue

            next_state, reward, done = env.step(action)

            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state, :] - q_table[state, action]))

            state = next_state
            total_rewards += reward

            if done:
                break

        epsilon *= np.exp(epsilon_decay * episode)

        rewards.append(total_rewards)

    return q_table, rewards


if __name__ == '__main__':
    mdp = PresentationEnvironment()

    # Hyperparameters
    episodes = 400
    steps_per_episode = 10
    alpha = 0.95            # learning rate
    epsilon = 1.            # exploration-exploitation rate
    gamma = 0.3             # discount rate

    q_table, rewards_over_time = q_learning(mdp, episodes, steps_per_episode, alpha, epsilon, gamma)

    optimal_policy = policy_iteration(mdp)
    q_policy = [np.argmax(q_table[s, :]) for s in range(mdp.number_of_states)]

    optimal_path, optimal_rewards = simulate(mdp, optimal_policy)
    q_path, q_rewards = simulate(mdp, q_policy)

    # Plot Rewards over episodes
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.plot(rewards_over_time, '-b', label='Q-learning')
    ax.plot([optimal_rewards] * len(rewards_over_time), '--r', label='Optimal solution')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Cumulative Rewards')

    plt.figtext(0.96, 0.02, 'alpha: {}; gamma: {}'.format(alpha, gamma), ha="right", fontsize=8)
    plt.legend()
    plt.show()

    print('Optimal Path: {}\nQ-Learning Path: {}'.format(' -> '.join(optimal_path), ' -> '.join(q_path)))