import numpy as np


class MDP(object):
    def __init__(self,
                 start_state: int,
                 states: [str],
                 states_actions: np.ndarray,
                 actions_transition_to_state: np.ndarray,
                 actions_rewards: np.ndarray
                 ):
        self.start_state = start_state
        self.states = states
        self.states_actions = states_actions
        self.actions_transition_to_state = actions_transition_to_state
        self.actions_rewards = actions_rewards

        assert 0 <= self.start_state < len(self.states), 'start_state must be an index of states'
        assert self.states_actions.shape[0] == len(self.states), 'states_actions row number must be the same as ' \
                                                                 'number of states'
        assert self.actions_rewards.shape[0] == self.actions_transition_to_state.shape[0]

    @property
    def number_of_states(self) -> int:
        return self.states_actions.shape[0]

    @property
    def number_of_actions(self) -> int:
        return self.actions_rewards.shape[0]


class WalkableMDP(MDP):

    def __init__(self, start_state: int, states: [str], states_actions: np.ndarray,
                 actions_transition_to_state: np.ndarray, actions_rewards: np.ndarray):
        super().__init__(start_state, states, states_actions, actions_transition_to_state, actions_rewards)
        self.current_state = self.start_state

    def reset(self):
        self.current_state = self.start_state

    def step(self, action):
        if action not in self.action_space:
            raise Exception('invalid action {}; available actions: {}'.format(action, self.action_space))

        reward = self.actions_rewards[action]
        next_state = np.random.choice(range(self.number_of_states), 1, p=self.actions_transition_to_state[action])[0]
        self.current_state = next_state
        done = len(self.states_actions[self.current_state]) == 0

        return self.current_state, reward, done

    @property
    def action_space(self):
        # return all possible actions of the current state
        return self.states_actions[self.current_state]


class PresentationEnvironment(WalkableMDP):
    # This is the reinforcement learning environment that I used in the presentation.

    def __init__(self):
        start_state = 2
        states = np.array(['Netflix', 'Sport', 'WA1', 'WA2', 'WA3', 'Sleep'])
        states_actions = np.array([
            [0],
            [1],
            [4, 6],
            [3, 5],
            [7],
            []
        ], dtype='object')
        actions_transition_to_state = np.array([
            [0.9, 0, 0.1, 0, 0, 0],
            [0.5, 0, 0, 0, 0.5, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
        ])
        actions_rewards = np.array([
            -5, 0, 0, 10, -2, -2, 10, 25
        ])

        super(PresentationEnvironment, self).__init__(start_state, states, states_actions,
                                                      actions_transition_to_state, actions_rewards)


def policy_iteration(env: MDP, gamma=0.8, epsilon=1.) -> [int]:
    values = [0] * env.number_of_states
    policy = [0] * env.number_of_states

    stable = False
    while not stable:
        values = policy_evaluation(env, policy, values, gamma, epsilon)
        policy, stable = policy_improvement(env, policy, values, gamma)

    return policy


def __set_policy_by_values(env, gamma, policy, state, values):
    best_state_value = float('-inf')
    for action in env.states_actions[state]:
        value = sum([p * (env.actions_rewards[action] + gamma * values[s2]) for s2, p in
                     enumerate(env.actions_transition_to_state[action])])
        if value > best_state_value:
            best_state_value = value
            policy[state] = action


def policy_evaluation(env: MDP, policy: [int], values: [int], gamma=0.8, epsilon=1.) -> [int]:
    while True:
        delta = 0

        for s in range(env.number_of_states):
            if len(env.states_actions[s]) == 0:
                continue

            v = values[s]

            action = policy[s]
            values[s] = sum([p * (env.actions_rewards[action] + gamma * values[s2]) for s2, p in
                             enumerate(env.actions_transition_to_state[action])])

            delta = max(delta, abs(v - values[s]))

        if delta < epsilon:
            break
    return values


def policy_improvement(env: MDP, policy: [int], values: [int], gamma=0.8) -> [int]:
    stable = True
    for s in range(env.number_of_states):
        old_action = policy[s]

        __set_policy_by_values(env, gamma, policy, s, values)

        if old_action != policy[s]:
            stable = False

    return policy, stable


def value_iteration(env: MDP, gamma=0.8, epsilon=1.) -> [int]:
    values = [0] * env.number_of_states

    while True:
        delta = 0.
        for s in range(env.number_of_states):
            if len(env.states_actions[s]) == 0:
                continue
            v = values[s]
            values[s] = np.max([sum([p * (env.actions_rewards[action] + gamma * values[s2]) for s2, p in
                                     enumerate(env.actions_transition_to_state[action])]) for action in
                                env.states_actions[s]])
            delta = max(delta, abs(v - values[s]))

        if delta < epsilon:
            break

    policy = [0] * env.number_of_states

    for s in range(env.number_of_states):
        __set_policy_by_values(env, gamma, policy, s, values)

    return policy


def simulate(env: WalkableMDP, policy: [int]) -> ([str], int):
    '''
    Simulate a complete episode by a given policy.
    '''
    assert env.number_of_states == len(policy), 'policy must have as many elements as the number of states!'

    env.reset()

    s = env.current_state
    done = False
    path = []
    total_rewards = 0

    while not done:
        path.append(env.states[s])
        s, r, done = env.step(policy[s])
        total_rewards += r
    path.append(env.states[s])

    return path, total_rewards


if __name__ == '__main__':

    mdp = PresentationEnvironment()

    print(policy_iteration(mdp))
    print(value_iteration(mdp))

    print(simulate(mdp, value_iteration(mdp)))