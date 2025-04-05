import numpy as np
import random
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt


class MultiArmedBandit:
    """
    Multi-Armed Bandit problem solver.

    This class implements several strategies for the multi-armed bandit problem,
    which is a simplified reinforcement learning setting with a single state.
    """

    def __init__(self, n_arms: int, true_rewards: Optional[List[float]] = None):
        """
        Initialize the Multi-Armed Bandit problem.

        Args:
            n_arms: Number of arms (actions)
            true_rewards: Optional list of true mean rewards for each arm
        """
        self.n_arms = n_arms

        if true_rewards is None:
            # Generate random true rewards if not provided
            self.true_rewards = np.random.normal(0, 1, n_arms)
        else:
            self.true_rewards = np.array(true_rewards)

        # Initialize estimates, counts, and total reward
        self.reset()

    def reset(self) -> None:
        """Reset all estimates and counts."""
        self.q_values = np.zeros(self.n_arms)  # Estimated value of each arm
        self.arm_counts = np.zeros(self.n_arms)  # Number of times each arm was pulled
        self.total_reward = 0  # Total reward received

    def pull_arm(self, arm: int) -> float:
        """
        Simulate pulling an arm and receiving a reward.

        Args:
            arm: The arm to pull

        Returns:
            The reward received
        """
        # Reward is drawn from a normal distribution centered at the true reward
        reward = np.random.normal(self.true_rewards[arm], 1.0)

        # Update estimates and counts
        self.arm_counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.arm_counts[arm]
        self.total_reward += reward

        return reward

    def epsilon_greedy(self, epsilon: float, n_steps: int) -> Tuple[List[float], List[int]]:
        """
        Run the epsilon-greedy algorithm.

        Args:
            epsilon: Exploration probability
            n_steps: Number of steps to run

        Returns:
            Tuple of (rewards, actions) for each step
        """
        self.reset()
        rewards = []
        actions = []

        for _ in range(n_steps):
            # Choose an action
            if random.random() < epsilon:
                # Explore: choose a random arm
                arm = random.randint(0, self.n_arms - 1)
            else:
                # Exploit: choose the arm with the highest estimated value
                arm = np.argmax(self.q_values)

            # Pull the arm and get reward
            reward = self.pull_arm(arm)

            rewards.append(reward)
            actions.append(arm)

        return rewards, actions

    def ucb(self, c: float, n_steps: int) -> Tuple[List[float], List[int]]:
        """
        Run the Upper Confidence Bound (UCB) algorithm.

        Args:
            c: Exploration parameter
            n_steps: Number of steps to run

        Returns:
            Tuple of (rewards, actions) for each step
        """
        self.reset()
        rewards = []
        actions = []

        # Pull each arm once to initialize
        for arm in range(self.n_arms):
            reward = self.pull_arm(arm)
            rewards.append(reward)
            actions.append(arm)

        for _ in range(n_steps - self.n_arms):
            # Choose an action using UCB
            t = np.sum(self.arm_counts)
            ucb_values = self.q_values + c * np.sqrt(np.log(t) / (self.arm_counts + 1e-10))
            arm = np.argmax(ucb_values)

            # Pull the arm and get reward
            reward = self.pull_arm(arm)

            rewards.append(reward)
            actions.append(arm)

        return rewards, actions

    def thompson_sampling(self, n_steps: int) -> Tuple[List[float], List[int]]:
        """
        Run the Thompson Sampling algorithm.

        Args:
            n_steps: Number of steps to run

        Returns:
            Tuple of (rewards, actions) for each step
        """
        self.reset()

        # Initialize Beta distribution parameters for each arm
        alpha = np.ones(self.n_arms)  # Successes plus 1 (prior)
        beta = np.ones(self.n_arms)  # Failures plus 1 (prior)

        rewards = []
        actions = []

        for _ in range(n_steps):
            # Sample from Beta distribution for each arm
            samples = [np.random.beta(alpha[i], beta[i]) for i in range(self.n_arms)]

            # Choose the arm with the highest sample
            arm = np.argmax(samples)

            # Pull the arm and get reward
            reward = self.pull_arm(arm)

            # Convert reward to binary (success/failure) for Beta distribution
            # Here we define success as reward > 0
            if reward > 0:
                alpha[arm] += 1
            else:
                beta[arm] += 1

            rewards.append(reward)
            actions.append(arm)

        return rewards, actions

    def plot_results(self, algorithm_results: Dict[str, Tuple[List[float], List[int]]]) -> None:
        """
        Plot the results of different algorithms.

        Args:
            algorithm_results: Dictionary mapping algorithm names to (rewards, actions) tuples
        """
        plt.figure(figsize=(15, 10))

        # Plot 1: Cumulative rewards
        plt.subplot(2, 1, 1)
        for name, (rewards, _) in algorithm_results.items():
            cumulative_rewards = np.cumsum(rewards)
            plt.plot(cumulative_rewards, label=name)

        plt.xlabel('Steps')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward Over Time')
        plt.legend()
        plt.grid(True)

        # Plot 2: Action counts
        plt.subplot(2, 1, 2)
        bar_width = 0.8 / len(algorithm_results)

        for i, (name, (_, actions)) in enumerate(algorithm_results.items()):
            action_counts = np.zeros(self.n_arms)
            for action in actions:
                action_counts[action] += 1

            x = np.arange(self.n_arms)
            plt.bar(x + i * bar_width, action_counts, width=bar_width, label=name)

        plt.xlabel('Arm')
        plt.ylabel('Number of Pulls')
        plt.title('Arm Selection Counts')
        plt.xticks(np.arange(self.n_arms) + bar_width * (len(algorithm_results) - 1) / 2)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


class ValueIteration:
    """
    Value Iteration algorithm for solving Markov Decision Processes (MDPs).

    Value Iteration is a dynamic programming algorithm that iteratively
    computes the optimal value function for each state in an MDP.
    """

    def __init__(self,
                 states: List[Any],
                 actions: List[Any],
                 transition_probs: Dict[Tuple[Any, Any, Any], float],
                 rewards: Dict[Tuple[Any, Any, Any], float],
                 gamma: float = 0.9,
                 theta: float = 1e-6):
        """
        Initialize the Value Iteration algorithm.

        Args:
            states: List of possible states
            actions: List of possible actions
            transition_probs: Dictionary mapping (state, action, next_state) to transition probability
            rewards: Dictionary mapping (state, action, next_state) to reward
            gamma: Discount factor (0 <= gamma <= 1)
            theta: Convergence threshold
        """
        self.states = states
        self.actions = actions
        self.transition_probs = transition_probs
        self.rewards = rewards
        self.gamma = gamma
        self.theta = theta

        self.state_indices = {state: i for i, state in enumerate(states)}
        self.action_indices = {action: i for i, action in enumerate(actions)}

        self.value_function = np.zeros(len(states))
        self.policy = {}

    def get_transition_prob(self, state: Any, action: Any, next_state: Any) -> float:
        """
        Get the transition probability for a state-action-next_state triplet.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state

        Returns:
            Transition probability P(next_state | state, action)
        """
        return self.transition_probs.get((state, action, next_state), 0.0)

    def get_reward(self, state: Any, action: Any, next_state: Any) -> float:
        """
        Get the reward for a state-action-next_state triplet.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state

        Returns:
            Reward R(state, action, next_state)
        """
        return self.rewards.get((state, action, next_state), 0.0)

    def run(self, max_iterations: int = 1000) -> Dict[Any, Any]:
        """
        Run the Value Iteration algorithm.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            The optimal policy as a dictionary mapping states to actions
        """
        # Initialize value function
        self.value_function = np.zeros(len(self.states))

        # Value iteration
        for i in range(max_iterations):
            delta = 0
            new_value_function = np.zeros_like(self.value_function)

            for s_idx, state in enumerate(self.states):
                # Calculate the value for each action
                action_values = []

                for action in self.actions:
                    value = 0

                    for next_state in self.states:
                        # Get transition probability and reward
                        prob = self.get_transition_prob(state, action, next_state)
                        reward = self.get_reward(state, action, next_state)

                        # Add to the expected value
                        next_state_idx = self.state_indices[next_state]
                        value += prob * (reward + self.gamma * self.value_function[next_state_idx])

                    action_values.append(value)

                # Set the new value to the maximum action value
                best_value = np.max(action_values) if action_values else 0
                new_value_function[s_idx] = best_value

                # Update delta for convergence check
                delta = max(delta, abs(new_value_function[s_idx] - self.value_function[s_idx]))

            # Update the value function
            self.value_function = new_value_function

            # Check for convergence
            if delta < self.theta:
                print(f"Value Iteration converged after {i + 1} iterations")
                break

        # Extract the policy
        self.policy = {}

        for state in self.states:
            self.state_indices[state]

            # Calculate the value for each action
            action_values = {}

            for action in self.actions:
                value = 0

                for next_state in self.states:
                    # Get transition probability and reward
                    prob = self.get_transition_prob(state, action, next_state)
                    reward = self.get_reward(state, action, next_state)

                    # Add to the expected value
                    next_state_idx = self.state_indices[next_state]
                    value += prob * (reward + self.gamma * self.value_function[next_state_idx])

                action_values[action] = value

            # Choose the action with the highest value
            best_action = max(action_values.items(), key=lambda x: x[1])[0] if action_values else None
            self.policy[state] = best_action

        return self.policy


class PolicyIteration:
    """
    Policy Iteration algorithm for solving Markov Decision Processes (MDPs).

    Policy Iteration alternates between policy evaluation (computing the value function
    for a fixed policy) and policy improvement (updating the policy based on the value function).
    """

    def __init__(self,
                 states: List[Any],
                 actions: List[Any],
                 transition_probs: Dict[Tuple[Any, Any, Any], float],
                 rewards: Dict[Tuple[Any, Any, Any], float],
                 gamma: float = 0.9,
                 theta: float = 1e-6):
        """
        Initialize the Policy Iteration algorithm.

        Args:
            states: List of possible states
            actions: List of possible actions
            transition_probs: Dictionary mapping (state, action, next_state) to transition probability
            rewards: Dictionary mapping (state, action, next_state) to reward
            gamma: Discount factor (0 <= gamma <= 1)
            theta: Convergence threshold
        """
        self.states = states
        self.actions = actions
        self.transition_probs = transition_probs
        self.rewards = rewards
        self.gamma = gamma
        self.theta = theta

        self.state_indices = {state: i for i, state in enumerate(states)}
        self.action_indices = {action: i for i, action in enumerate(actions)}

        # Initialize with a random policy
        self.policy = {state: np.random.choice(actions) if actions else None for state in states}
        self.value_function = np.zeros(len(states))

    def get_transition_prob(self, state: Any, action: Any, next_state: Any) -> float:
        """
        Get the transition probability for a state-action-next_state triplet.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state

        Returns:
            Transition probability P(next_state | state, action)
        """
        return self.transition_probs.get((state, action, next_state), 0.0)

    def get_reward(self, state: Any, action: Any, next_state: Any) -> float:
        """
        Get the reward for a state-action-next_state triplet.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state

        Returns:
            Reward R(state, action, next_state)
        """
        return self.rewards.get((state, action, next_state), 0.0)

    def policy_evaluation(self, max_iterations: int = 100) -> np.ndarray:
        """
        Evaluate the current policy by computing its value function.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            The value function for the current policy
        """
        # Initialize value function
        value_function = np.zeros(len(self.states))

        # Iterate until convergence or max iterations
        for i in range(max_iterations):
            delta = 0

            for s_idx, state in enumerate(self.states):
                v = value_function[s_idx]

                # For the action prescribed by the policy
                action = self.policy[state]

                # Calculate the new value
                new_value = 0

                for next_state in self.states:
                    # Get transition probability and reward
                    prob = self.get_transition_prob(state, action, next_state)
                    reward = self.get_reward(state, action, next_state)

                    # Add to the expected value
                    next_state_idx = self.state_indices[next_state]
                    new_value += prob * (reward + self.gamma * value_function[next_state_idx])

                value_function[s_idx] = new_value
                delta = max(delta, abs(v - value_function[s_idx]))

            # Check for convergence
            if delta < self.theta:
                break

        return value_function

    def policy_improvement(self, value_function: np.ndarray) -> Tuple[Dict[Any, Any], bool]:
        """
        Improve the policy based on the value function.

        Args:
            value_function: The value function for the current policy

        Returns:
            Tuple of (new_policy, policy_stable) where:
            - new_policy: The improved policy
            - policy_stable: Whether the policy has converged
        """
        policy_stable = True
        new_policy = {}

        for state in self.states:
            old_action = self.policy[state]

            # Calculate the value for each action
            action_values = {}

            for action in self.actions:
                value = 0

                for next_state in self.states:
                    # Get transition probability and reward
                    prob = self.get_transition_prob(state, action, next_state)
                    reward = self.get_reward(state, action, next_state)

                    # Add to the expected value
                    next_state_idx = self.state_indices[next_state]
                    value += prob * (reward + self.gamma * value_function[next_state_idx])

                action_values[action] = value

            # Choose the action with the highest value
            best_action = max(action_values.items(), key=lambda x: x[1])[0] if action_values else None
            new_policy[state] = best_action

            # Check if the policy has changed
            if old_action != best_action:
                policy_stable = False

        return new_policy, policy_stable

    def run(self, max_iterations: int = 100) -> Dict[Any, Any]:
        """
        Run the Policy Iteration algorithm.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            The optimal policy as a dictionary mapping states to actions
        """
        for i in range(max_iterations):
            # Policy evaluation
            self.value_function = self.policy_evaluation()

            # Policy improvement
            new_policy, policy_stable = self.policy_improvement(self.value_function)

            # Update the policy
            self.policy = new_policy

            # Check for convergence
            if policy_stable:
                print(f"Policy Iteration converged after {i + 1} iterations")
                break

        return self.policy


class QLearning:
    """
    Q-Learning algorithm for reinforcement learning.

    Q-Learning is a model-free algorithm that learns the value of an action
    in a given state (Q-value) without requiring a model of the environment.
    """

    def __init__(self,
                 states: List[Any],
                 actions: List[Any],
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.1):
        """
        Initialize the Q-Learning algorithm.

        Args:
            states: List of possible states
            actions: List of possible actions
            alpha: Learning rate (0 < alpha <= 1)
            gamma: Discount factor (0 <= gamma <= 1)
            epsilon: Exploration rate (0 <= epsilon <= 1)
        """
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.state_indices = {state: i for i, state in enumerate(states)}
        self.action_indices = {action: i for i, action in enumerate(actions)}

        # Initialize Q-table
        self.q_table = np.zeros((len(states), len(actions)))

        # Keep track of visited states and the best policy
        self.visited_states = set()
        self.policy = {}

    def choose_action(self, state: Any) -> Any:
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            The chosen action
        """
        state_idx = self.state_indices[state]

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # Exploitation (choose best action)
        action_idx = np.argmax(self.q_table[state_idx])
        return self.actions[action_idx]

    def update(self, state: Any, action: Any, reward: float, next_state: Any) -> None:
        """
        Update the Q-value for a state-action pair.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        state_idx = self.state_indices[state]
        action_idx = self.action_indices[action]
        next_state_idx = self.state_indices[next_state]

        # Q-learning update
        best_next_q = np.max(self.q_table[next_state_idx])
        self.q_table[state_idx, action_idx] += self.alpha * (
                reward + self.gamma * best_next_q - self.q_table[state_idx, action_idx]
        )

        # Mark state as visited
        self.visited_states.add(state)

    def get_policy(self) -> Dict[Any, Any]:
        """
        Get the current greedy policy based on the Q-table.

        Returns:
            Dictionary mapping states to actions
        """
        self.policy = {}

        for state in self.visited_states:
            state_idx = self.state_indices[state]
            action_idx = np.argmax(self.q_table[state_idx])
            self.policy[state] = self.actions[action_idx]

        return self.policy

    def train(self, env, n_episodes: int = 1000, max_steps: int = 100) -> Tuple[Dict[Any, Any], List[float]]:
        """
        Train the Q-learning agent on an environment.

        Args:
            env: Environment with step(action) method that returns (next_state, reward, done)
            n_episodes: Number of episodes to train for
            max_steps: Maximum steps per episode

        Returns:
            Tuple of (final_policy, rewards_per_episode)
        """
        rewards_per_episode = []

        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0

            for step in range(max_steps):
                # Choose an action
                action = self.choose_action(state)

                # Take the action and observe the next state and reward
                next_state, reward, done = env.step(action)

                # Update the Q-value
                self.update(state, action, reward, next_state)

                state = next_state
                total_reward += reward

                if done:
                    break

            rewards_per_episode.append(total_reward)

            # Decay epsilon over time (exploration rate)
            self.epsilon = max(0.01, self.epsilon * 0.99)

        # Get the final policy
        final_policy = self.get_policy()

        return final_policy, rewards_per_episode


class SARSA:
    """
    SARSA (State-Action-Reward-State-Action) algorithm for reinforcement learning.

    SARSA is an on-policy algorithm that learns the value of the policy being followed
    (unlike Q-learning which learns the value of the optimal policy).
    """

    def __init__(self,
                 states: List[Any],
                 actions: List[Any],
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.1):
        """
        Initialize the SARSA algorithm.

        Args:
            states: List of possible states
            actions: List of possible actions
            alpha: Learning rate (0 < alpha <= 1)
            gamma: Discount factor (0 <= gamma <= 1)
            epsilon: Exploration rate (0 <= epsilon <= 1)
        """
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.state_indices = {state: i for i, state in enumerate(states)}
        self.action_indices = {action: i for i, action in enumerate(actions)}

        # Initialize Q-table
        self.q_table = np.zeros((len(states), len(actions)))

        # Keep track of visited states and the best policy
        self.visited_states = set()
        self.policy = {}

    def choose_action(self, state: Any) -> Any:
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            The chosen action
        """
        state_idx = self.state_indices[state]

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # Exploitation (choose best action)
        action_idx = np.argmax(self.q_table[state_idx])
        return self.actions[action_idx]

    def update(self, state: Any, action: Any, reward: float, next_state: Any, next_action: Any) -> None:
        """
        Update the Q-value for a state-action pair.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action to be taken
        """
        state_idx = self.state_indices[state]
        action_idx = self.action_indices[action]
        next_state_idx = self.state_indices[next_state]
        next_action_idx = self.action_indices[next_action]

        # SARSA update
        next_q = self.q_table[next_state_idx, next_action_idx]
        self.q_table[state_idx, action_idx] += self.alpha * (
                reward + self.gamma * next_q - self.q_table[state_idx, action_idx]
        )

        # Mark state as visited
        self.visited_states.add(state)

    def get_policy(self) -> Dict[Any, Any]:
        """
        Get the current greedy policy based on the Q-table.

        Returns:
            Dictionary mapping states to actions
        """
        self.policy = {}

        for state in self.visited_states:
            state_idx = self.state_indices[state]
            action_idx = np.argmax(self.q_table[state_idx])
            self.policy[state] = self.actions[action_idx]

        return self.policy

    def train(self, env, n_episodes: int = 1000, max_steps: int = 100) -> Tuple[Dict[Any, Any], List[float]]:
        """
        Train the SARSA agent on an environment.

        Args:
            env: Environment with step(action) method that returns (next_state, reward, done)
            n_episodes: Number of episodes to train for
            max_steps: Maximum steps per episode

        Returns:
            Tuple of (final_policy, rewards_per_episode)
        """
        rewards_per_episode = []

        for episode in range(n_episodes):
            state = env.reset()
            action = self.choose_action(state)
            total_reward = 0

            for step in range(max_steps):
                # Take the action and observe the next state and reward
                next_state, reward, done = env.step(action)

                # Choose the next action
                next_action = self.choose_action(next_state)

                # Update the Q-value
                self.update(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action
                total_reward += reward

                if done:
                    break

            rewards_per_episode.append(total_reward)

            # Decay epsilon over time (exploration rate)
            self.epsilon = max(0.01, self.epsilon * 0.99)

        # Get the final policy
        final_policy = self.get_policy()

        return final_policy, rewards_per_episode