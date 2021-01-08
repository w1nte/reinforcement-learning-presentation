from presentation_example_env import EnvDP, simulate
import matplotlib.pyplot as plt
import numpy as np
import random


env = EnvDP()

q_table = np.zeros((env.number_of_states, env.number_of_actions))

# Plot
rewards = []

# Hyperparameters
alpha = 0.9  # learning rate
epsilon = 1.  #
episodes = 400
gamma = 0.9
steps_per_episode = 30

# Policy Training
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

    epsilon *= np.exp(-0.0001 * episode)
    rewards.append(total_rewards)

env2 = EnvDP()
optimal_path, optimal_rewards = simulate(env2, env2.policy_iteration(0.01))

# Plot Rewards over episodes
plt.plot(rewards, label='Q-learning')
plt.plot([optimal_rewards] * len(rewards), label='Optimal Solution')
plt.xlabel('Episodes')
plt.ylabel('Sum of Rewards')
plt.legend()
plt.show()

env.reset()
path = [env.states[env.current_state]]
done = False
while not done:
    action = np.argmax(q_table[env.current_state, :])
    _, _, done = env.step(action)
    path.append(env.states[env.current_state])

print(' -> '.join(path))