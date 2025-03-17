import torch.nn as nn
import torch
from typing import List

from src.storage.episode_buffer import EpisodeBuffer, EpisodeStep


class NeuralNetManager:
    def __init__(
        self,
        representation: nn.Module,
        dynamics: nn.Module,
        prediction: nn.Module,
        learning_rate=0.01,
    ):
        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction
        self.learning_rate = learning_rate
        self.training_steps = 0

    def NNr(self):
        # TODO implement method with in/out
        pass

    def NNd(self):
        # TODO implement method with in/out
        pass

    def NNp(self, state):
        return self.prediction(state)

    def __lr_decay(self, decay_rate=0.95):
        return self.learning_rate * decay_rate**self.training_steps

    def translate_and_evaluate(self, game_state):
        # Translate game state to hidden state
        hidden_state = self.representation(game_state)

        # Evaluate hidden state
        policy, value = self.prediction(hidden_state)

        # Only translate, so reward is 0 (tensor)
        reward = torch.tensor(0, dtype=torch.float32)

        policy_p = policy[0]

        return hidden_state, value, reward, [p for p in policy_p], policy

    def transition_and_evaluate(self, hidden_state, action):
        # Transition hidden state with action
        next_hidden_state, reward = self.dynamics(hidden_state, action)

        # Evaluate hidden state
        policy, value = self.prediction(next_hidden_state)

        policy_p = policy[0]

        return next_hidden_state, value, reward, [p for p in policy_p], policy

    def get_weights(self) -> List[List[torch.Tensor]]:
        networks = [self.representation, self.dynamics, self.prediction]
        return [net.get_parameters() for net in networks]

    def bptt(self, buffer: EpisodeBuffer, q: int, K: int = 5) -> torch.Tensor:
        batch = buffer.sample_state(q, K)

        lr = self.__lr_decay()
        optimizer = torch.optim.Adam(self.get_weights(), lr=lr, weight_decay=1e-4)

        loss = self.__update_weights(batch, optimizer, q)

        self.training_steps += 1

        return loss

    def __update_weights(
        self, batch: list[EpisodeStep], optimizer: torch.optim.Optimizer, q: int
    ) -> torch.Tensor:
        # Assume network is an instance of your MuZero network with proper submodules.
        # optimizer is a PyTorch optimizer, e.g., torch.optim.SGD or Adam.
        # optimizer.zero_grad()  # Clear gradients

        # Translate to hidden state
        state_seq = [step.state for step in batch[0:q]]
        prev_action_seq = [step.action for step in batch[0:q]]

        action_seq = [step.action for step in batch[q + 1 :]]
        policy_seq = [step.policy for step in batch[q + 1 :]]
        reward_seq = [step.reward for step in batch[q + 1 :]]
        value_seq = [step.value for step in batch[q + 1 :]]

        optimizer.zero_grad()
        loss = torch.tensor(0, dtype=torch.float32)
        targets = [
            (state.value, state.reward, state.policy)
            for state in zip(value_seq, reward_seq, policy_seq)
        ]

        # --- Initial Inference Step ---
        # Forward pass for the initial observation.
        representation_input = torch.stack([state_seq, prev_action_seq], dim=1)
        hidden_state, value, reward, _, policy_t = self.translate_and_evaluate(
            representation_input
        )
        predictions = [(1.0, value, reward, policy_t)]

        for action in action_seq:
            hidden_state, value, reward, _, policy_t = self.transition_and_evaluate(
                hidden_state, action
            )
            predictions.append((1.0 / len(action_seq), value, reward, policy_t))

            hidden_state = self.__scale_gradient(hidden_state, 0.5)
            hidden_state = self.__scale_gradient(hidden_state, 0.5)

        for k, (prediction, target) in enumerate(zip(predictions, targets)):
            gradient_scale, value, reward, policy_t = prediction
            target_value, target_reward, target_policy = target

            l_v = self.__loss_fn(value, [target_value])

            l_r = 0
            l_p = 0

            if k > 0:
                l_r = self.__loss_fn(reward, [target_reward]).to(torch.float32)
                l_p = self.__policy_loss_fn([target_policy], policy_t)

            l_loss = l_r + l_v + l_p

            loss += self.__scale_gradient(l_loss, gradient_scale)

        loss /= len(predictions)

        loss.backward()
        optimizer.step()

        return loss

    def __scale_gradient(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        # Scales the gradient for the backward pass.
        return tensor * scale + tensor.detach() * (1 - scale)

    def __loss_fn(
        self, value: torch.Tensor, target_value: torch.Tensor
    ) -> torch.Tensor:
        loss_fn = torch.nn.MSELoss()
        return loss_fn(value, target_value)

    def __policy_loss_fn(
        self, policy_logits: torch.Tensor, target_policy: torch.Tensor
    ) -> torch.Tensor:
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(policy_logits, target_policy)
