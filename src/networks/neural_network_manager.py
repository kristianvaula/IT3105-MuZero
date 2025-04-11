import numpy as np
import torch
import torch.nn as nn
from typing import List
from src.networks.neural_network import RepresentationNetwork
from src.networks.neural_network import DynamicsNetwork
from src.networks.neural_network import PredictionNetwork


from src.storage.episode_buffer import EpisodeBuffer, EpisodeStep


class NeuralNetManager:
    def __init__(
        self,
        representation: RepresentationNetwork,
        dynamics: DynamicsNetwork,
        prediction: PredictionNetwork,
        learning_rate=0.01,
    ):
        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction
        self.learning_rate = learning_rate
        self.training_steps = 0

    def NNr(self, input):
        return self.representation(input)
    
    def NNd(self, state, action):
        """
        Dynamics network: predicts next state and reward given current state and action.
        
        Args:
            state: Current abstract state representation (tensor)
            action: Action to take (integer)
            
        Returns:
            Tuple of (next_state, reward)
        # Convert action to tensor if it's not already
        if not isinstance(action, torch.Tensor):
            action = torch.tensor([action], dtype=torch.float32)
        
        # Handle different input shapes
        if len(state.shape) == 0:  # Scalar tensor
            state = state.unsqueeze(0)
        
        # Ensure action has the right shape for concatenation
        if action.dim() < state.dim():
            action = action.unsqueeze(0)
        
        # Concatenate state and action
        combined_input = torch.cat((state, action), dim=-1)
        
        # Use the dynamics network which should return both next_state and reward
        return self.dynamics(combined_input)

        # TODO: REVERT TO OLD NNd CODE IF EDVARD FUCKS UP
        """
        
        # TODO: The dynamics network only handles linear layers as the first layer: 
        # check if first layer is linear
        if not isinstance(action, torch.Tensor):
            action = torch.tensor([action], dtype=torch.long)
        # Call the dynamics network with the current hidden state and the action.
        next_hidden_state, reward = self.dynamics(state, action)
        return next_hidden_state, reward


    def NNp(self, state):
        return self.prediction(state)

    def __lr_decay(self, decay_rate=0.99):
        return self.learning_rate * decay_rate**self.training_steps

    def translate_and_evaluate(self, game_state):
        """
        Given the raw game state (an image tensor of shape (C, H, W)),
        first ensure it has a batch dimension, then produce its corresponding
        hidden representation, policy logits, and value estimate.
        Initial reward is always set to zero.
        """
        if not isinstance(game_state, torch.Tensor):
            game_state = torch.tensor(game_state, dtype=torch.float32)
        # Add batch dimension if missing.
        if game_state.dim() == 3:
            game_state = game_state.unsqueeze(0)

        hidden_state = self.representation(game_state)
        policy_logits, value = self.prediction(hidden_state)
        reward = torch.tensor([0], dtype=torch.float32)
        policy = torch.softmax(policy_logits, dim=-1)
        return hidden_state, value, reward, policy_logits, policy

    def transition_and_evaluate(self, hidden_state, action):
        """
        Given the current hidden state and an action, use the dynamics network to get the next hidden state
        and reward, and then evaluate the new hidden state with the prediction network.
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor([action], dtype=torch.long)
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_hidden_state)
        policy = torch.softmax(policy_logits, dim=-1)
        return next_hidden_state, value, reward, policy_logits, policy

    def get_weights(self) -> List[torch.Tensor]:
        networks = [self.representation, self.dynamics, self.prediction]
        # Each network is assumed to have a method get_parameters() that returns its parameters.
        return [p for net in networks for p in net.get_parameters()]

    def bptt(self, buffer: EpisodeBuffer, q: int, w: int, minibatch_size: int) -> torch.Tensor:
        """
        q: number of steps to look back (state window)
        w: number of steps to look forward from 0
        minibatch_size: number of episodes to sample
        """
        batch = buffer.sample(q, w, minibatch_size)
        lr = self.__lr_decay()
        optimizer = torch.optim.Adam(self.get_weights(), lr=lr, weight_decay=1e-4)

        loss_history = []
        lr_history = [lr] * len(batch)
        for episode in batch:
            loss_history.append(self.__update_weights(episode, optimizer, q))
            self.training_steps += 1

        return loss_history, lr_history

    def __update_weights(
        self, batch: list[EpisodeStep], optimizer: torch.optim.Optimizer, q: int
    ) -> torch.Tensor:
        optimizer.zero_grad()  # Clear gradients

        # For Atari, we take the state from the last element in the history window.
        state_seq = np.array([step.state for step in batch[0:q]], dtype=np.float32)
        # For actions, assume we store them as integers.
        prev_action_seq = np.array([step.action for step in batch[0:q]], dtype=np.float32)

        action_seq = [step.action for step in batch[q + 1:]]
        policy_seq = [step.policy for step in batch[q + 1:]]
        reward_seq = [step.reward for step in batch[q + 1:]]
        value_seq = [step.value for step in batch[q + 1:]]

        loss = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        targets = [(v, r, p) for v, r, p in zip(value_seq, reward_seq, policy_seq)]

        # --- Initial Inference Step ---
        # Use only the most recent state in the history. For Atari, that state is an image.
        state_tensor = torch.tensor(state_seq[-1], dtype=torch.float32).unsqueeze(0)
        hidden_state, value, reward, _, policy_logits = self.translate_and_evaluate(state_tensor)
        # Save the prediction along with an initial gradient scale of 1.0.
        predictions = [(1.0, value, reward, policy_logits)]

        # Unroll the rollout using actions in action_seq.
        for i, action in enumerate(action_seq):
            hidden_state, value, reward, policy_logits, _ = self.transition_and_evaluate(
                hidden_state, action
            )
            gradient_scale = 1.0 / (np.log(i + 2) + 1)  # Weight decays with unroll length.
            predictions.append((gradient_scale, value, reward, policy_logits))
            # Optionally scale gradients in the hidden state to control the impact of long unrolls.
            hidden_state = self.__scale_gradient(hidden_state, 0.5)

        # Compute loss from predictions vs. targets.
        for k, (prediction, target) in enumerate(zip(predictions, targets)):
            gradient_scale, value, reward, policy_logits = prediction
            target_value, target_reward, target_policy = target

            target_value = torch.tensor(target_value, dtype=torch.float32).unsqueeze(0)
            target_reward = torch.tensor(target_reward, dtype=torch.float32).unsqueeze(0)
            target_policy = torch.tensor(target_policy, dtype=torch.float32)

            l_v = self.__loss_fn(value, target_value)
            l_r = self.__loss_fn(reward, target_reward) if k > 0 else 0
            l_p = self.__policy_loss_fn(policy_logits, target_policy) if k > 0 else 0

            l_loss = l_v + l_r + l_p
            loss = loss + self.__scale_gradient(l_loss, gradient_scale)

        loss = loss / len(predictions)
        loss.backward()
        optimizer.step()

        return loss

    def __scale_gradient(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        # Scales the gradient flowing backwards.
        return tensor * scale + tensor.detach() * (1 - scale)

    def __loss_fn(self, value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
        return nn.MSELoss()(value, target_value)

    def __policy_loss_fn(
        self, policy_logits: torch.Tensor, target_policy: torch.Tensor
    ) -> torch.Tensor:
        log_probs = torch.log_softmax(policy_logits, dim=-1)
        return nn.functional.kl_div(log_probs, target_policy, reduction="batchmean")