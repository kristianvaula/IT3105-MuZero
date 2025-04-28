import random
import numpy as np
import torch
from src.gsm.gsm import GameStateManager
from src.self_play.uMCTS import uMCTS
from src.networks.neural_network_manager import NeuralNetManager
from src.storage.episode_buffer import EpisodeBuffer
from src.config import Config
from src.storage.episode_buffer import Episode
import wandb
import os
import torchvision.transforms as T

class ReinforcementLearningManager:
    """
    Reinforcement Learning Manager (RLM) for MuZero.

    Runs the main training algorithm:
      1. Runs episodes in the environment.
      2. At each time step, gathers a history of real-game states (ϕₖ),
          computes the abstract state (σₖ) using the representation network (NNr),
          and then performs u-MCTS search from that state.
      3. Collects training data (state, value, policy, action, reward) for each time step.
      4. Every training_interval episodes, performs BPTT training using data from the episode buffer.
    """

    def __init__(
        self,
        env,
        gsm: GameStateManager,
        monte_carlo: uMCTS,
        nnm: NeuralNetManager,
        episode_buffer: EpisodeBuffer,
        config: Config,
    ):
        self.env = env
        self.gsm = gsm
        self.monte_carlo = monte_carlo
        self.nnm = nnm
        self.episode_buffer = episode_buffer
        self.config = config

        self.num_episodes = (
            config.environment.num_episodes
        )  # Ne: total number of episodes
        self.num_episode_steps = (
            config.environment.num_episode_step
        )  # Nes: max steps per episode
        self.training_interval = (
            config.environment.training_interval
        )  # It: training update interval
        self.minibatch_size = (
            config.environment.minibatch_size
        )  # mbs: minibatch size for BPTT
        self.history_length = (
            config.networks.state_window
        )  # q: number of past states required for φₖ

        self.roll_ahead = (
            config.networks.roll_ahead
        )
        # w: number of steps to roll ahead in the environment

        self.use_wandb = (
            config.logging.use_wandb if hasattr(config.logging, "use_wandb") else False
        )
        if self.use_wandb:
            wandb.init(
                project=config.logging.wandb_project,
                entity="IT3105-muzero",
                name=config.logging.wandb_name,
                config=vars(config) if hasattr(config, "__dict__") else config,
            )

    def train(self):
        """
        Runs the full training loop and returns the trained NeuralNetManager (nnm).
        """
        counter = 0
        best_reward = -float("inf")
        checkpoint_interval = (
            self.config.logging.checkpoint_interval
            if hasattr(self.config.logging, "checkpoint_interval")
            else 100
        )

        for episode in range(self.num_episodes):
            # (a) Reset env and obtain initial state s₀.
            current_state, _ = self.env.reset()
            action = self.env.action_space.sample()

            episode_data = Episode()

            state_history = []
            action_history = []
            done = False

            episode_reward = 0
            episode_steps = 0

            for _ in range(self.num_episode_steps):
                state_history.append(current_state)
                action_history.append(action)
                phi_k_tensor = self._get_phi_k(state_history, action_history)
                abstract_state, _, _, _, _ = self.nnm.translate_and_evaluate(
                    phi_k_tensor
                )
                
                # Initialize u-Tree with abtract state σ₀ and run uMCTS
                policy, root_value = self.monte_carlo.search(abstract_state)
                # Sample action aₖ₊₁ from the policy πₖ
                action = self._sample_action(policy)

                # Simulate one timestep in the game: sₖ₊₁, rₖ₊₁
                next_state, reward, done, terminated, _ = self.env.step(action)

                episode_reward += reward
                episode_steps += 1

                # Save the training data for this step.
                episode_data.add_step(
                    state=current_state,
                    value=root_value.item(), # TODO: Incorrect Target Value Calculation?
                    policy=list(policy.values()), 
                    action=action,
                    reward=reward,
                )

                current_state = next_state
                if done or terminated:
                    break

            # Add the completed episode to the episode buffer
            self.episode_buffer.add_episode(episode_data)

            if self.use_wandb:
                wandb.log(
                    {
                        "episode": episode,
                        "episode_reward": episode_reward,
                        "episode_steps": episode_steps,
                        "episode_final_step_value_estimate": root_value.item()
                    }
                )

                if (episode + 1) % checkpoint_interval == 0:
                    self.save_checkpoint(episode + 1)

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    if episode % 10:
                        self.save_checkpoint(f"{episode + 1}_best")

            # Every training_interval episodes, perform BPTT training
            if (episode + 1) % self.training_interval == 0:
                # TODO: Networks are only trained every training_interval episodes:
                # This might be too infrequent for efficient learning, depending on your specific environment.
                loss_history, lr_history = self.nnm.bptt(self.episode_buffer, self.history_length, self.roll_ahead, self.minibatch_size)
                # loss is tensor so print as list of values
                loss_history = loss_history.tolist() if isinstance(loss_history, torch.Tensor) else loss_history
                lr_history = lr_history.tolist() if isinstance(lr_history, torch.Tensor) else lr_history
                if self.use_wandb:
                    for loss, lr in zip(loss_history, lr_history):
                        counter +=1
                        wandb.log({
                            "training_step": counter,
                            "loss": round(float(loss), 4),
                            "learning_rate": round(float(lr), 4),
                        })
                


                print(f"Episode {episode + 1} Loss: {[round(float(loss), 4) for loss in loss_history]}")
                
        if self.use_wandb:
            wandb.finish()

        return self.nnm

    def play(self, num_episodes: int):
        """
        Plays the game using the trained NeuralNetManager (nnm) for a given number of episodes.
        Renders the environment at each time step.
        """
        for ep in range(num_episodes):
            print(f"\n--- Starting Episode {ep + 1} ---")
            current_state, _ = self.env.reset()
            done = False

            # Reset history for the current episode
            state_history = []
            action_history = []

            # Optionally render the initial state
            if hasattr(self.env, "render"):
                self.env.render()

            while not done and len(state_history) < self.num_episode_steps:
                # Append current state to history
                state_history.append(current_state)

                # If there are no actions yet, you can choose a default (or sample a random one)
                # so that the history tensor is correctly built
                if not action_history:
                    action_history.append(self.env.action_space.sample())

                # Create the state-action history tensor (ϕₖ)
                phi_k_tensor = self._get_phi_k(state_history, action_history)

                # Obtain the abstract state from the trained networks.
                abstract_state, _, _, _, _ = self.nnm.translate_and_evaluate(
                    phi_k_tensor
                )

                # Run u-MCTS starting from the abstract state.
                policy, root_value = self.monte_carlo.search(abstract_state)

                # Sample the next action from the computed policy.
                action = self._select_action(policy)
                # action = self._sample_action(policy)
                action_history.append(action)

                # Take a step in the environment.
                current_state, reward, done, terminated, info = self.env.step(action)

                # Optionally render the environment.
                if hasattr(self.env, "render"):
                    
                    self.env.render(reward=reward==1)

            print("Episode finished!")



    def _get_phi_k(self, state_history, action_history):
        """
        Returns the history of states ϕₖ.
        """
        if self.config.environment_name == "snakepac":
            state_window = self.history_length
            state_tensor = torch.tensor(
                np.array(state_history[-state_window:]), dtype=torch.float32
            ).flatten()
            action_tensor = torch.tensor(
                action_history[-state_window:], dtype=torch.float32
            )
            cat = torch.cat((state_tensor, action_tensor), dim=0)
            return cat
        elif self.config.environment_name == "riverraid":
            window = self.history_length  # Should be 32 according to config
            
            if len(state_history) < window:
                pad_length = window - len(state_history)
                state_history = [state_history[0]] * pad_length + state_history
                action_history = [self.env.action_space.sample()] * pad_length + action_history 
            
            # Take the last 'window' states and actions
            recent_states = state_history[-window:]
            recent_actions = action_history[-window:]
            
            # Create tensor for RGB states (32 frames × 3 channels × 96×96)
            # Note: Skip the grayscale conversion to keep RGB channels
            rgb_transform = T.Compose([
                T.ToPILImage(),
                T.Resize((96, 96)),
                T.ToTensor()  # Returns [C, H, W] format with values [0, 1]
            ])
            
            state_tensors = []
            for state in recent_states:
                # Transform to 3×96×96 (keeping RGB)
                state_tensor = rgb_transform(state)  # shape: [3, 96, 96]
                state_tensors.append(state_tensor)
            
            # Stack all state tensors: [32, 3, 96, 96]
            states_tensor = torch.stack(state_tensors)
            
            # Reshape to [32*3, 96, 96] = [96, 96, 96]
            states_tensor = states_tensor.view(-1, 96, 96)
            
            # Create action bias planes [32, 96, 96]
            action_tensors = []
            for action in recent_actions:
                # Normalize action value (assuming action space of 18 as in config)
                norm_action = float(action) / 18.0
                # Create a constant plane filled with this action value
                action_plane = torch.full((96, 96), norm_action, dtype=torch.float32)
                action_tensors.append(action_plane)
            
            # Stack all action planes: [32, 96, 96]
            actions_tensor = torch.stack(action_tensors)
            
            # Combine state and action tensors to form input with 128 channels
            # [96+32, 96, 96] = [128, 96, 96]
            phi = torch.cat([states_tensor, actions_tensor], dim=0)
            
            return phi
            
        else:
            raise NotImplementedError("Only SnakePac environment is supported.")

    def _sample_action(self, policy):
        """
        Samples an action from the policy.
        """
        actions = list(policy.keys())
        probabilities = list(policy.values())
        return random.choices(actions, weights=probabilities, k=1)[0]
    
    def _select_action(self, policy):
        """
        Selects the action with the highest probability from the policy.
        """
        actions = list(policy.keys())
        probabilities = list(policy.values())
        return actions[np.argmax(probabilities)]
    

    def save_checkpoint(self, episode):
        """
        Saves model checkpoint and logs it to wandb.
        """
        os.makedirs("checkpoints", exist_ok=True)

        checkpoint_path = f"checkpoints/model_ep{episode}.pt"
        self.nnm.representation.save_model()
        self.nnm.dynamics.save_model()
        self.nnm.prediction.save_model()

        if self.use_wandb:
            wandb.save(checkpoint_path)
