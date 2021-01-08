import numpy as np

class Env(object):
    # This a reinforcement learning environment that I used in the presentation.

    def __init__(self):
        self.states = np.array([
            'Netflix', 'Sport', 'WA1', 'WA2', 'WA3', 'Sleep'
        ])

        # state to action
        self.state_actions = np.array([
            [0],
            [1],
            [4, 6],
            [3, 5],
            [7],
            []
        ], dtype='object')

        # transition matrix action to state
        self.actions_transition_to_state = np.array([
            [0.9, 0, 0.1, 0, 0, 0],
            [0.5, 0, 0, 0, 0.5, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
        ])

        # rewards matrix action
        self.actions_rewards = np.array([
            -5, 0, 0, 10, -2, -2, 10, 25
        ])

        self.number_of_states = len(self.states)
        self.number_of_actions = len(self.actions_rewards)

        self.current_state = 2
        self.reset()

    def reset(self):
        self.current_state = 2

    def step(self, action):
        if action not in self.action_space:
            raise Exception('invalid action; available actions: {}'.format(self.action_space))

        reward = self.actions_rewards[action]
        next_state = np.random.choice(range(6), 1, p=self.actions_transition_to_state[action])[0]
        self.current_state = next_state
        done = self.current_state == 5
        return self.current_state, reward, done

    @property
    def action_space(self):
        # return all actions of the current state
        return self.state_actions[self.current_state]


class EnvDP(Env):

    gamma = 0.8

    def __init__(self):
        super().__init__()

    def policy_iteration(self, epsilon=1.):
        values = [0] * self.number_of_states
        policy = [0] * self.number_of_states

        stable = False
        while not stable:
            values = self.policy_evaluation(policy, values, epsilon)
            policy, stable = self.policy_improvement(policy, values)
            print(values, policy, stable)

        return policy

    def policy_evaluation(self, policy, values, epsilon=1.):
        while True:
            delta = 0

            for s in range(self.number_of_states):
                if len(self.state_actions[s]) == 0:
                    continue

                v = values[s]

                action = policy[s]
                values[s] = sum([p * (self.actions_rewards[action] + self.gamma * values[s2]) for s2, p in enumerate(self.actions_transition_to_state[action])])

                delta = max(delta, abs(v - values[s]))

            if delta < epsilon:
                break
        return values

    def policy_improvement(self, policy, values):
        stable = True
        for s in range(self.number_of_states):
            old_action = policy[s]

            best_state_value = float('-inf')
            for action in self.state_actions[s]:
                value = sum([p * (self.actions_rewards[action] + self.gamma * values[s2]) for s2, p in enumerate(self.actions_transition_to_state[action])])
                if value > best_state_value:
                    best_state_value = value
                    policy[s] = action

            if old_action != policy[s]:
                stable = False

        return policy, stable

    def value_iteration(self, epsilon=1.):
        values = [0] * self.number_of_states

        while True:
            delta = 0.
            for s in range(self.number_of_states):
                if len(self.state_actions[s]) == 0:
                    continue
                v = values[s]
                values[s] = np.max([sum([p * (self.actions_rewards[action] + self.gamma * values[s2]) for s2, p in enumerate(self.actions_transition_to_state[action])]) for action in self.state_actions[s]])
                delta = max(delta, abs(v - values[s]))

            if delta < epsilon:
                break

        print(values)

        policy = [0] * self.number_of_states
        for s in range(self.number_of_states):
            best_state_value = float('-inf')
            for action in self.state_actions[s]:
                value = sum([p * (self.actions_rewards[action] + self.gamma * values[s2]) for s2, p in
                             enumerate(self.actions_transition_to_state[action])])
                if value > best_state_value:
                    best_state_value = value
                    policy[s] = action
        return policy


def simulate(env, policy):
    s = env.current_state
    done = False
    path = []
    total_rewards = 0
    while not done:
        path += env.states[s]
        s, r, done = env.step(policy[s])
        total_rewards += r
    path += env.states[s]

    return path, total_rewards


if __name__ == '__main__':
    env = EnvDP()

    policy = env.policy_iteration(0.01)
    policy2 = env.value_iteration(0.01)
    print(policy)
    print(policy2)

    s = env.current_state
    done = False
    while not done:
        print(env.states[s])
        s, r, done = env.step(policy2[s])
    print(env.states[s])