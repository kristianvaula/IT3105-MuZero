import random
import numpy as np
import torch
from src.gsm.gsm import GameStateManager
from src.self_play.uMCTS import uMCTS
from src.networks.neural_network_manager import NeuralNetManager
from src.storage.episode_buffer import EpisodeBuffer
from src.config import Config
from src.storage.episode_buffer import Episode
from src.wrappers.single_life_wrapper import SingleLifeWrapper
import wandb
import os
import torchvision.transforms as T
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import time

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
                loss_history, lr_history = self.nnm.bptt(self.episode_buffer, self.history_length, self.roll_ahead, self.minibatch_size, self.config.environment)
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

    def play(self, num_episodes: int, record_folder: str | None = None):
        """
        Plays the game using the trained model. Optionally records video.
        This version aims to keep the core MCTS/NN logic identical while adding recording.

        Args:
            num_episodes: Number of episodes to play.
            record_folder: If provided, saves videos of episodes to this folder.
                           Requires ffmpeg and moviepy (pip install moviepy).
                           Set to None to disable recording and use existing env rendering.
        """
        print(f"\n--- Starting Playback ---")
        print(f"Playing {num_episodes} episodes.")

        can_render_human_base = hasattr(self.env, 'render_mode') and self.env.render_mode == 'human'

        if record_folder:
            print(f"Recording enabled. Videos will be saved to: {record_folder}")
            os.makedirs(record_folder, exist_ok=True)
            if not can_render_human_base:
                 print("Note: Base environment wasn't in 'human' mode, display window won't be shown.")
        elif can_render_human_base:
            print("Recording disabled. Using existing environment for playback with rendering.")
        else:
            print("Recording disabled. Running playback without rendering.")


        if self.nnm.representation.network is None: # Basic check
             print("ERROR: Representation network seems uninitialized. Cannot play.")
             return


        for ep in range(num_episodes):
            print(f"\n--- Starting Episode {ep + 1}/{num_episodes} ---")

            env_to_run = None
            is_recording_this_episode = False

            try:
                if record_folder:
                    env_id = f"ALE/{self.config.environment_name.capitalize()}-v5" # Adjust if not ALE
                    print(f"  Setting up environment '{env_id}' for single-life recording...")
                    play_env_base = gym.make(env_id, render_mode="rgb_array")
                    play_env_single_life = SingleLifeWrapper(play_env_base)
                    _, initial_info = play_env_single_life.reset(seed=self.config.environment.seed + ep)
                    print(f"  Initial lives for recording ep {ep+1}: {initial_info.get('lives', 'N/A')}")

                    env_to_run = RecordVideo(
                        play_env_single_life,
                        video_folder=record_folder,
                        episode_trigger=lambda x: True, 
                        name_prefix=f"muzero-play-ep{ep+1}",
                        disable_logger=True
                    )
                    is_recording_this_episode = True
                    print(f"  Recording episode {ep+1} using dedicated rgb_array environment.")
                else:
                    print("  Using original environment instance for playback.")
                    _, initial_info = self.env.reset(seed=self.config.environment.seed + ep)
                    env_to_run = self.env
                    is_recording_this_episode = False

            except Exception as e:
                 print(f"  ERROR setting up environment for episode {ep+1}: {e}")
                 import traceback
                 traceback.print_exc()
                 print(f"  Skipping episode {ep+1}.")
                 continue 


            try:
                current_state, _ = env_to_run.reset()
                done = False
                terminated = False

                state_history = [current_state]
                action = env_to_run.action_space.sample()
                action_history = [action]

                episode_steps = 0
                total_reward = 0

                while not (done or terminated) and episode_steps < self.num_episode_steps:

                    phi_k_tensor = self._get_phi_k(state_history, action_history)
                    phi_k_tensor = phi_k_tensor.to(self.nnm.representation.device)

                    with torch.no_grad():
                        abstract_state, _, _, _, _ = self.nnm.translate_and_evaluate(phi_k_tensor)

                        policy, root_value = self.monte_carlo.search(abstract_state)

                    action = self._select_action(policy)

                    current_state, reward, terminated, done, info = env_to_run.step(action)
                    total_reward += reward

                    state_history.append(current_state)
                    action_history.append(action) 
                    episode_steps += 1

                    if not is_recording_this_episode and can_render_human_base:
                        if env_to_run == self.env:
                             env_to_run.render()

                print(f"Episode {ep + 1} finished. Steps: {episode_steps}, Reward: {total_reward:.2f}")

            except Exception as e:
                 print(f"  ERROR during episode {ep+1} execution: {e}")
                 import traceback
                 traceback.print_exc()
            finally:
                if env_to_run is not None:
                    env_to_run.close()
                    if is_recording_this_episode:
                        print(f"  Video recording finalized for episode {ep+1}.")

        print("\n--- Playback Finished ---")



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
    

    def save_checkpoint(self, episode): # episode can be int or string like "XXX_best"
        """
        Saves model checkpoint components into a subdirectory named after the episode/tag.
        """
        save_dir = "checkpoints" # Save checkpoints here for clarity
        # Use the 'episode' identifier (int or string) as the subdirectory name
        checkpoint_subdir = str(episode)
        full_save_dir_path = os.path.join(save_dir, checkpoint_subdir) # Path to the specific checkpoint directory

        print(f"Saving checkpoint for '{episode}' to directory: {full_save_dir_path}...")
        # Note: The underlying save_model will create the directory if it doesn't exist.

        try:
            # Pass the specific subdirectory name and base directory to each save_model call
            self.nnm.representation.save_model(
                subdir=checkpoint_subdir,
                model_name="representation_model.pth",
                dir=save_dir
            )
            self.nnm.dynamics.save_model(
                subdir=checkpoint_subdir,
                model_name="dynamics_model.pth",
                dir=save_dir
            )
            self.nnm.prediction.save_model(
                subdir=checkpoint_subdir,
                model_name="prediction_model.pth",
                dir=save_dir
            )
            print(f"  Checkpoint files saved successfully for '{episode}'.")

            if self.use_wandb:
                 try:
                     artifact_name = f"model_checkpoint_{episode}" 
                     artifact = wandb.Artifact(artifact_name, type='model',
                                               description=f"MuZero Model Checkpoint for {episode}")

                     rep_path = os.path.join(full_save_dir_path, "representation_model.pth")
                     dyn_path = os.path.join(full_save_dir_path, "dynamics_model.pth")
                     prd_path = os.path.join(full_save_dir_path, "prediction_model.pth")

                     if os.path.exists(rep_path): artifact.add_file(rep_path, name="representation_model.pth")
                     if os.path.exists(dyn_path): artifact.add_file(dyn_path, name="dynamics_model.pth")
                     if os.path.exists(prd_path): artifact.add_file(prd_path, name="prediction_model.pth")


                     wandb.log_artifact(artifact)
                     print(f"  Logged checkpoint artifact to wandb: {artifact_name}")
                 except Exception as e:
                     print(f"  WARN: Failed to log checkpoint artifact to wandb: {e}")

        except Exception as e:
            print(f"  ERROR during checkpoint saving process for '{episode}': {e}")
            import traceback
            traceback.print_exc() 