import torch
import random
import numpy as np
from gymnasium import Env
from src.self_play.uMCTS import uMCTS
from src.storage.episode_buffer import EpisodeBuffer, EpisodeHistory


class ReinforcementLearningModule:
    def __init__(
        self,
        umcts: uMCTS,
        env: Env,
        nnm,
        buffer_size=10000,
        batch_size=64,
        discount_factor=0.99,
        lr=0.001,
    ):
        """
        Initializes the Reinforcement Learning Module (RLM).
        """
        self.env = env
        self.umcts = umcts  # u-MCTS for planning
        self.nnm = nnm  # neural network manager, used for NNr, NNd, NNp
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.discount_factor = discount_factor

        self.episode_buffer = EpisodeBuffer()
        self.episode_history = EpisodeHistory()
        self.episode_history.load_history()
        self.q = 5  # Number of real-game states to gather for each training step

        self.optimizer = torch.optim.Adam(
            list(self.nnm.nnr.parameters())
            + list(self.nnm.nnd.parameters())
            + list(self.nnm.nnp.parameters()),
            lr=lr,
        )  # TODO: Recheck when NNM is finished. Also replace with NNM

    def collect_experience(self, num_simulations=10):
        """
        Self-play loop using MCTS to generate training data.
        """
        for _ in range(num_simulations):
            state, _ = self.env.reset()
            done = False
            game_data = []

            while not done:
                mcts_policy, _ = self.mcts.search(state)
                action = random.choices(
                    list(mcts_policy.keys()), weights=mcts_policy.values()
                )[0]
                next_state, reward, done, _, _ = self.env.step(action)
                game_data.append((state, action, reward, next_state, mcts_policy, done))
                state = next_state

            self.episode_buffer.add_episode(game_data)
            self.episode_history.add_episode(game_data)

    def __train_bptt(self, minibatch_size):
        """
        Performs Backpropagation Through Time (BPTT) to train Ψ.

        Args:
            minibatch_size: Number of episodes to sample for training.
        """
        if len(self.episode_history.history) < minibatch_size:
            return None  # Not enough data to train

        batch = self.episode_history.sample_episodes(minibatch_size)

        states, values, policies, actions, rewards = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        values = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1)
        policies = torch.tensor(np.array(policies), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)

        predicted_values = self.nnm.nnp(states)[
            1
        ]  # TODO: Recheck when NNM is finished. Also replace with NNM
        predicted_policies = self.nnm.nnp(states)[
            0
        ]  # TODO: Recheck when NNM is finished. Also replace with NNM
        predicted_rewards = self.nnm.nnd(states, actions)[
            1
        ]  # TODO: Recheck when NNM is finished. Also replace with NNM

        value_loss = torch.nn.functional.mse_loss(predicted_values, values)
        policy_loss = -(policies * predicted_policies.log()).sum(dim=1).mean()
        reward_loss = torch.nn.functional.mse_loss(predicted_rewards, rewards)

        total_loss = value_loss + policy_loss + reward_loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        print(f"BPTT Training Loss: {total_loss.item():.4f}")

    def __train_step(self, state, num_simulations):
        # ϕk = {sk−q ,...,sk }# Gather q+1 real-game states. Fill with blank states when k<q.

        history = 0
        if len(self.episode_buffer) < self.q:
            # fill with blank states
            history = 0  # TODO: fill with blank states
        else:
            history = self.episode_history.sample_episodes(self.q)

        # σk = NNr (ϕk ) # Create the abstract state
        # s = self.nnm.nnr(state)
        abstract_state = self.nnm.nnr(
            history
        )  # TODO: Recheck when NNM is finished. Also replace with NNM

        self.umcts.search(abstract_state)

        #  Initialize u-Tree root with abstract state σk.

        for _ in range(num_simulations):
            self.umcts.search(abstract_state)

        # πk = normalized distribution of visit counts in u-Tree along all arcs emanating from root.

        # v∗k = value of the root node

        policy, root_value = self.umcts.search(abstract_state)

        # Sample action ak+1 from πk

        action = np.random.choice(len(policy), p=policy)

        # s_k+1,r^∗_k+1 = Simulate game one timestep(s_k ,a_k+1) # Get new state and reward

        next_state, reward, done, _, _ = self.env.step(action)

        # episode_data.append([s_k ,v^∗_k ,π_k ,a_k+1,r^∗_k+1])

        episode_data = (state, root_value, policy, action, reward)
        return next_state, done, episode_data

    def __train_episode(self, steps, num_simulations):
        # Reset the video game to an initial state (s0)
        state, _ = self.env.reset()
        done = False
        episode_data = []

        # epidata ←∅# episode data (used for training Ψ)
        for _ in range(steps):
            if done:
                break

            state, done, step_data = self.__train_step(state, num_simulations)

            episode_data.append(step_data)

        # EH.append(epidata) # Add the episode data to the episode history

        self.episode_history.add_episode(episode_data)

        # if episode modulo I_t == 0:
        # DO BPTT TRAINING(Ψ,EH,mbs)

    def train(
        self,
        epochs=100,
        steps=100,
        num_simulations=10,
        update_interval=10,
        minibatch_size=64,
    ):
        """
        Run the reinforcement learning training loop.

        # Initialize episode history

        # For episode in range of num_episodes:
        # Reset the video game to an initial state (s0)

        # Initialize episode data

        # For k in range num_simulations:
            # Gather q+1 real game states. Fill blank states when k < q.

            # Createa a new abstract state (s) using the NNr.

            # Perform u-MCTS search starting from the current abstract state (s)

            # Select an action using the policy returned by u-MCTS search

            # Take the selected action in the video game and observe the next state (s')

            # Store the transition (s, a, r, s') in the episode data

            # If the game has ended, break

        # Add the episode data to the episode history

        # If episode modulo I_t == 0:
            # Back propagation through time (BPTT) using trident, EH, and MBS

        # Return network parameters
        """

        # EH ←∅# Episode History = all training data from all episodes.

        # Randomly initialize parameters (weights and biases) of Ψ

        # For episode in range(Ne):

        for epoch in range(epochs):
            self.__train_episode(steps, num_simulations)

            if (epoch + 1) % update_interval == 0:
                self.__train_bptt(minibatch_size)

            if (epoch + 1) % update_interval == 0:
                self.episode_history.save_history()

        #  Return Ψ # The fully-trained neural networks: NNr ,NNd,andNNp

        return self.nnm  # TODO: Recheck when NNM is finished. Also replace with NNM
