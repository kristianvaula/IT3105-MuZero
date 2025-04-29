import torch
import torch.nn as nn
from typing import List
from src.storage.episode_buffer import EpisodeBuffer, EpisodeStep
from src.networks.neural_network import RepresentationNetwork
from src.networks.neural_network import DynamicsNetwork
from src.networks.neural_network import PredictionNetwork
import torchvision.transforms as T





class NeuralNetManager:
    def __init__(
        self,
        representation: RepresentationNetwork,
        dynamics: DynamicsNetwork,
        prediction: PredictionNetwork,
        learning_rate=0.001,
    ):
        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction
        self.learning_rate = learning_rate
        self.training_steps = 0

    _rgb_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((96, 96)),
        T.ToTensor()
    ])
    

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

        # TODO: REVERT TO OLD NNd CODE IF FUCKS UP
        """
        
        # TODO: The dynamics network only handles linear layers as the first layer: 
        # check if first layer is linear
        if not isinstance(action, torch.Tensor):
            action = torch.tensor([action], dtype=torch.long, device=state.device)
        elif action.dtype != torch.long:
            action = action.to(dtype=torch.long, device=state.device)

        if state.dim() > 1 and action.dim() == 1:
            # If state is a batch of states and action is a single action, expand action
            action = action.squeeze(0)

        # Call the dynamics network with the current hidden state and the action.
        next_hidden_state, reward = self.dynamics(state, action)
        return next_hidden_state, reward


    def NNp(self, state):
        return self.prediction(state)

    def __lr_decay(self, decay_rate=0.99):
        return max(self.learning_rate * decay_rate**self.training_steps, 1e-4)

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

        game_state = game_state.to(self.representation.device)

        hidden_state = self.representation(game_state)
        policy_logits, value = self.prediction(hidden_state)
        reward = torch.tensor([0.0], dtype=torch.float32, device=hidden_state.device)
        policy = torch.softmax(policy_logits, dim=-1)
        return hidden_state, value, reward, policy_logits, policy

    def transition_and_evaluate(self, hidden_state, action):
        """
        Given the current hidden state and an action, use the dynamics network to get the next hidden state
        and reward, and then evaluate the new hidden state with the prediction network.
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor([action], dtype=torch.long, device=hidden_state.device)

        elif action.dtype != torch.long:
            action = action.to(dtype=torch.long, device=hidden_state.device)

        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_hidden_state)
        policy = torch.softmax(policy_logits, dim=-1)
        reward = reward.to(device=hidden_state.device)
        return next_hidden_state, value, reward, policy_logits, policy

    def get_weights(self) -> List[torch.nn.Parameter]:
        return self.representation.get_parameters() + \
               self.dynamics.get_parameters() + \
               self.prediction.get_parameters()

    def _prepare_initial_riverraid_input(self, state_history, action_history, window_size, action_space_size):
        """
        Helper function to prepare the initial 128-channel input tensor for Riverraid,
        based on the logic from rlm.py's _get_phi_k.
        """
        # Ensure history has the required window size (padding if necessary)
        if len(state_history) < window_size:
            pad_length = window_size - len(state_history)
            # Pad with the oldest available state/action
            state_history = [state_history[0]] * pad_length + state_history
            # Use a reasonable default padding action if necessary, like 0 or a random sample
            # Assuming action_history might be shorter or need padding too.
            # Let's assume action_history also needs padding similar to state_history logic.
            # If action_history[0] might not exist, use a default like 0.
            padding_action = action_history[0] if action_history else 0
            action_history = [padding_action] * pad_length + action_history

        # Take the last 'window_size' states and actions
        recent_states = state_history[-window_size:]
        recent_actions = action_history[-window_size:]

        # Create tensor for RGB states
        state_tensors = [self._rgb_transform(state) for state in recent_states] # shape: list of [3, 96, 96]
        states_tensor = torch.stack(state_tensors) # shape: [window_size, 3, 96, 96]

        # Reshape to [window_size*3, 96, 96] (e.g., [96, 96, 96] for window_size=32)
        states_tensor = states_tensor.view(-1, 96, 96)

        # Create action bias planes
        action_tensors = []
        for action in recent_actions:
            # Normalize action value
            norm_action = float(action) / float(action_space_size) # Use actual action space size
            action_plane = torch.full((96, 96), norm_action, dtype=torch.float32)
            action_tensors.append(action_plane)

        actions_tensor = torch.stack(action_tensors) # shape: [window_size, 96, 96]

        # Combine state and action tensors: [channels, H, W]
        # e.g., [96+32, 96, 96] = [128, 96, 96]
        phi = torch.cat([states_tensor, actions_tensor], dim=0)

        # Ensure the tensor is on the correct device (if using CUDA)
        # Assuming self.representation.device holds the target device ('cuda' or 'cpu')
        phi = phi.to(self.representation.device)

        return phi


    def bptt(self, buffer: EpisodeBuffer, q: int, w: int, minibatch_size: int, env_config) -> torch.Tensor:
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
            if len(episode) < q + w:
                continue
            loss_tensor = self.__update_weights(episode, optimizer, q, w, env_config)
            loss_history.append(loss_tensor)
            self.training_steps += 1 

        if not loss_history:
            return torch.tensor([]), torch.tensor([])
        
        loss_tensor_combined = torch.stack(loss_history)

        lr_tensor_combined = torch.tensor(lr_history[:len(loss_history)]) # Match length if some skipped


        return loss_tensor_combined, lr_tensor_combined

    def __update_weights(
        self, batch: list[EpisodeStep], optimizer: torch.optim.Optimizer, q: int, w: int, env_config
    ) -> torch.Tensor:
        optimizer.zero_grad()  # Clear gradients

        # For Atari, we take the state from the last element in the history window.
        state_history = [step.state for step in batch[0:q]]
        action_history = [step.action for step in batch[0:q]] 
        
        # prev_action_seq = np.array([step.action for step in batch[0:q]], dtype=np.float32)

        action_seq_future = [step.action for step in batch[q : q + w]]

        policy_seq_targets = [step.policy for step in batch[q : q + w]]
        reward_seq_targets = [step.reward for step in batch[q : q + w]] # r_{q+1} to r_{q+w}
        value_seq_targets = [step.value for step in batch[q : q + w]]   # v_q to v_{q+w-1}

        actual_w = len(action_seq_future)
        if actual_w == 0:
             print(f"Warning: No future steps available in batch segment after q={q}. Skipping update.")
             return torch.tensor(0.0, device=self.representation.device) # Return zero loss tensor


        w = actual_w 

        targets = list(zip(value_seq_targets, reward_seq_targets, policy_seq_targets))

        initial_input_tensor = self._prepare_initial_riverraid_input(
            state_history, action_history, q, env_config.action_space # Use q for window size
        ) # Shape [128, 96, 96], on correct device

        hidden_state, value, reward_pred_dummy, policy_logits, _ = self.translate_and_evaluate(initial_input_tensor)

        predictions = [(1.0, value, reward_pred_dummy, policy_logits)]

        # Unroll the rollout using actions in action_seq.
        for i, action in enumerate(action_seq_future):
            hidden_state, value, reward_pred, policy_logits, _ = self.transition_and_evaluate(
                hidden_state, action
            )
            # gradient_scale = 1.0 / w # Or some other scheme
            gradient_scale = 1.0 # Often scale=1 is used here
            predictions.append((gradient_scale, value, reward_pred, policy_logits))
            # Scale gradient flowing back through hidden state
            hidden_state = self.__scale_gradient(hidden_state, 0.5)

        total_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=value.device)

        # Compute loss from predictions vs. targets.
        for k in range(w):
            gradient_scale, pred_value, pred_reward_k, pred_policy_logits = predictions[k] # Prediction for state k
            target_value_k, target_reward_kp1, target_policy_k = targets[k] # Targets related to state k and action k

            # Ensure targets are tensors and on the correct device
            device = pred_value.device
            target_value_tensor = torch.tensor([target_value_k], dtype=torch.float32, device=device)
            target_reward_tensor = torch.tensor([target_reward_kp1], dtype=torch.float32, device=device) # Reward r_{q+k+1}
            target_policy_tensor = torch.tensor(target_policy_k, dtype=torch.float32, device=device)
            if target_policy_tensor.dim() == 1:
                target_policy_tensor = target_policy_tensor.unsqueeze(0) # Add batch dim if needed

            # Value loss: Compare predicted value at step k with MCTS value at step k
            l_v = self.__loss_fn(pred_value, target_value_tensor)

            # Reward loss: Compare predicted reward for step k+1 with actual reward r_{q+k+1}
            # Prediction[k+1] contains reward resulting from action a_{q+k} (transition k -> k+1)
            # Target[k] contains reward r_{q+k+1}
            # We need prediction k+1's reward here.
            _, _, pred_reward_kp1, _ = predictions[k+1] # Reward predicted for transition k -> k+1
            l_r = self.__loss_fn(pred_reward_kp1, target_reward_tensor)

            # Policy loss: Compare predicted policy at step k with MCTS policy at step k
            l_p = self.__policy_loss_fn(pred_policy_logits, target_policy_tensor)

            # Combine losses for this step
            step_loss = l_v + l_r + l_p

            # Scale and accumulate loss
            # total_loss = total_loss + self.__scale_gradient(step_loss, gradient_scale) # Apply scaling if needed
            total_loss = total_loss + step_loss # Simpler: add step losses

        if w > 0:
            total_loss = total_loss / w
        else:
            # Avoid division by zero if w is 0 (should have been caught earlier)
            total_loss = torch.tensor(0.0, device=value.device)

        if total_loss.requires_grad: # Check if loss requires grad before backward()
             total_loss.backward()
             torch.nn.utils.clip_grad_norm_(self.get_weights(), max_norm=1.0)
             optimizer.step()
        else:
             print("Warning: Loss does not require grad. Skipping backward/step.")

        return total_loss.detach()

    def __scale_gradient(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        # Scales the gradient flowing backwards.
        return tensor * scale + tensor.detach() * (1 - scale)

    def __loss_fn(self, value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
        if value.shape != target_value.shape:
            target_value = target_value.view_as(value)
        return nn.MSELoss()(value, target_value)

    def __policy_loss_fn(
        self, policy_logits: torch.Tensor, target_policy: torch.Tensor
    ) -> torch.Tensor:
        # Ensure target has the same shape [Batch, Num_Actions]
        if policy_logits.shape != target_policy.shape:
            target_policy = target_policy.view_as(policy_logits)

        # Ensure target_policy is a valid probability distribution (non-negative, sums to 1)
        # Clamp target policy probabilities just in case MCTS produces slightly off values
        target_policy = torch.clamp(target_policy, min=0)
        target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True)

        # CrossEntropyLoss expects raw logits and target probabilities
        # Use reduction='mean' to average over the batch
        return nn.CrossEntropyLoss(reduction='mean')(policy_logits, target_policy)