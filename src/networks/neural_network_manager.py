import torch.nn as nn
import torch

from src.storage.episode_buffer import EpisodeBuffer

class NeuralNetManager():
    def __init__(self, representation: nn.Module, dynamics: nn.Module, prediction: nn.Module, learning_rate=0.01):
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
         
    def get_weights(self):
        networks = [self.representation, self.dynamics, self.prediction]
        return [net.get_parameters() for net in networks]
    
    def bptt(self, buffer: EpisodeBuffer, q, K=5):
        batch = buffer.sample_state(q)
        
        lr = self.__lr_decay()
        optimizer = torch.optim.Adam(self.get_weights(), lr=lr)
        
        loss = self.__update_weights(batch, optimizer)
        
        self.training_steps += 1
        
        return loss
        
    def __update_weights(self, batch, optimizer):
        # Assume network is an instance of your MuZero network with proper submodules.
        # optimizer is a PyTorch optimizer, e.g., torch.optim.SGD or Adam.
            # optimizer.zero_grad()  # Clear gradients
        # --- Initial Inference Step ---
        # Forward pass for the initial observation.
        optimizer.zero_grad()
        loss = 0
        actions = [state.action for state in batch]
        targets = [(state.value, state.reward, state.policy) for state in batch]
        
        for step in batch:
             # Initial step, from the real observation.
            hidden_state, value, reward, _, policy_t = self.translate_and_evaluate(step.state)
            predictions = [(1.0, value, reward, policy_t)]

            # Recurrent steps, from action and previous hidden state.
            for action in actions:
        
                hidden_state, value, reward, _, policy_t = self.transition_and_evaluate(hidden_state, action)
                predictions.append((1.0 / len(actions), value, reward, policy_t))

                hidden_state = self.__scale_gradient(hidden_state, 0.5)

            for k, (prediction, target) in enumerate(zip(predictions, targets)):
        
                gradient_scale, value, reward, policy_t = prediction
                target_value, target_reward, target_policy = target

                l_a = self.__loss_fn(value, [target_value])
            
                l_b = 0
                l_c = 0
                
                if k > 0:
                    l_b = self.__loss_fn(reward, [target_reward]).to(torch.float32)
                    l_c = self.__policy_loss_fn([target_policy], policy_t)
                
                l =  l_a + l_b + l_c       
            
                loss += self.__scale_gradient(l, gradient_scale)
                
        loss /= len(batch)
        
        loss.backward()
        optimizer.step()
        
        return loss
                
    
    def __scale_gradient(self, tensor, scale: float):
        # Scales the gradient for the backward pass.
        return tensor * scale + tensor.detach() * (1 - scale)

    def __loss_fn(self, value, target_value):
        loss_fn = torch.nn.MSELoss()
        return loss_fn(value, target_value)

    def __policy_loss_fn(self, policy_logits, target_policy):
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(policy_logits, target_policy)

    
    
    
