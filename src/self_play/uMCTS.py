# u-MCTS
# – Performs Monte Carlo Tree Search in the abstract state space.
# – Uses the ASM and NNM to expand nodes, perform rollouts, and backpropagate value estimates.
import math
import random
from typing import Dict
from src.gsm.gsm import GameStateManager


class Node:
    def __init__(self, abstract_state, parent=None):
        self.state = abstract_state
        self.parent = parent
        self.children: Dict[int, Node] = {}  # dict: action -> child Node
        self.visit_count = 0
        self.q_value = 0.0  # Estimated cumulative reward
        self.reward = 0.0  # Reward from parent's action


class uMCTS:
    """
    Monte Carlo Tree Search in the abstract state space.
    """

    def __init__(
        self,
        nnm,
        gsm: GameStateManager,
        action_space,
        num_searches,
        max_depth,
        ucb_constant,
        discount_factor,
    ):
        self.nnm = nnm
        self.gsm = gsm
        self.action_space = action_space

        self.num_searches = num_searches
        self.max_depth = max_depth
        self.ucb_constant = ucb_constant
        self.discount_factor = discount_factor

    def search(self, root_state):
        """
        Perform u-MCTS search starting from the given root abstract state.
        """
        root = Node(root_state)

        for _ in range(self.num_searches):
            node = root
            # -------------------------------
            # TREE POLICY: Traverse from root to a leaf
            # -------------------------------
            while not self.__is_leaf(node):
                node = self.__select_child(node)

            # -------------------------------
            # EXPANSION: If we haven't reached max depth, expand the leaf node
            # (u-MCTS always expands up to a fixed maximum depth)
            # -------------------------------
            if self.__depth(node) < self.max_depth:
                self.__expand(node)

            # For rollout, if the node has children, pick one at random; else use the node itself
            rollout_node = (
                random.choice(list(node.children.values())) if node.children else node
            )

            # -------------------------------
            # ROLLOUT: Simulate actions from rollout_node for the remaining depth using NNd and NNp
            # -------------------------------
            accum_reward = self.__rollout(
                rollout_node, self.max_depth - self.__depth(rollout_node)
            )

            # -------------------------------
            # BACKPROPAGATION: Update Q-values and visit counts along the path
            # -------------------------------
            self.__backpropagate(rollout_node, accum_reward)

        # After simulations, compute the probability distribution over actions
        policy = self.__compute_policy(root)

        _, root_value = self.nnm.NNp(root.state)
        return policy, root_value

    def __is_leaf(self, node: Node):
        """Check if the node has children to determine if it is a leaf node"""
        return len(node.children) == 0

    def __select_child(self, node: Node):
        """Select a child node using an UCB policy."""
        best_score = -float("inf")
        best_child = None
        for _, child in node.children.items():
            c = self.ucb_constant
            # Calculte Upper Confidence Bound score
            score = child.q_value + c * math.sqrt(
                math.log(node.visit_count + 1) / (child.visit_count + 1)
            )
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def __depth(self, node: Node):
        """Compute the depth of the node in the tree."""
        depth = 0
        while node.parent is not None:
            node = node.parent
            depth += 1
        return depth

    def __expand(self, node: Node):
        """
        Expand the given node by generating one child for each possible action.
        For the root node, legal actions can be obtained vis GSM.
        For the deeper nodes, we assume all actions are possible.
        """
        if node.parent is None:
            actions = self.gsm.get_legal_actions(node.state)
        else:
            actions = self.action_space

        for action in actions:
            if action not in node.children:
                next_state, predicted_reward = self.nnm.NNd(node.state, action)
                child_node = Node(next_state, parent=node)
                child_node.reward = predicted_reward
                node.children[action] = child_node

    def __rollout(self, node: Node, remaining_depth):
        """
        Perform a rollout from the given node for a fixed depth.
        At each step, use the prediction network (NNp) to obtain a policy and value.
        Then sample an action from the predicted policy and use NNd to get the next state and reward.
        """
        accum_reward = 0
        discount = 1.0
        current_state = node.state
        for d in range(remaining_depth):
            policy, _ = self.nnm.NNp(current_state)
            action = self.__sample_action(policy)

            next_state, reward = self.nnm.NNd(current_state, action)
            accum_reward += discount * reward
            discount *= self.discount_factor
            current_state = next_state

        _, final_value = self.nnm.NNp(current_state)
        accum_reward += discount * final_value
        return accum_reward

    def __sample_action(self, policy):
        """
        Sample an action from a probability distribution.
        """
        actions = list(policy.keys())
        probabilities = list(policy.values())
        return random.choices(actions, weights=probabilities, k=1)[0]

    def __backpropagate(self, node: Node, accum_reward):
        """
        Backpropagate the rollout reward up the tree, updating visit counts and Q-values.
        """
        while node is not None:
            node.visit_count += 1
            node.q_value = (
                node.q_value * (node.visit_count - 1) + accum_reward
            ) / node.visit_count
            node = node.parent

    def __compute_policy(self, root: Node):
        """
        Compute a probability distribution over actions based on the visit
        counts of the root's children.
        """
        total_visits = sum(child.visit_count for child in root.children.values())
        policy = {
            action: child.visit_count / total_visits
            for action, child in root.children.items()
        }
        return policy
