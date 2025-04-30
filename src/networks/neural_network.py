import torch
from src.networks.network_builder import NetworkBuilder
import os


def build_head(head_config):
    if type(head_config) is list:
        layers = []
        for config in head_config:
            if config is not None:
                built_layer_list = NetworkBuilder([config]).build_layer(config)
                if built_layer_list:
                     layers.append(built_layer_list[0])
    else:
        layers = NetworkBuilder([head_config]).build_layer(head_config)
        if not isinstance(layers, list):
             layers = [layers] if layers else []
        
    return torch.nn.Sequential(*layers) 


class NeuralNetwork(torch.nn.Module):
    def __init__(self, layer_configs, device, build=True, iteration=None):
        """
        Build a network from layer configurations.
        Each configuration is a dict that includes:
          - "type": type of the layer ("linear", "conv2d", etc.)
          - Required parameters (e.g., in_features, out_features).
          - Optional "activation": Activation function name (e.g., "relu").
        """
        super().__init__() 
        self.layer_configs = layer_configs
        self.device = device

        if layer_configs:
             self.network = NetworkBuilder(layer_configs).build_network()
        else:
            self.network = None


    def __call__(self, x, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    def save_model(
        self,
        subdir: int | None = None,
        model_name="representation_model.pth",
        dir="models",
    ):
        """
        Save the model to a directory.

        Args:
            subdir (str): Subdirectory to save the model - > UNIX timestamp as int.
            model_name (str): Name of the model file, with type.
            dir (str): Directory to save the model.
        """
        if not hasattr(self, 'state_dict'): # Basic check
             raise ValueError("Model cannot be saved, state_dict method missing.")


        if self.network is None:
            raise ValueError("Model has not been initialized. Cannot save.")

        if not os.path.exists(dir):
            os.makedirs(dir)

        if subdir is None:
            subdirs = os.listdir(dir)
            if not subdirs:
                subdir = 0
            else:
                subdir = str(max(int(s) for s in subdirs))
        else:
            subdir = str(subdir)

        dir_path = f"{dir}/{subdir}"
        os.makedirs(dir_path, exist_ok=True)

        model_path = f"{dir_path}/{model_name}"
        #torch.save(self.network.state_dict(), model_path)
        torch.save(self.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def load_model(
        self, iteration=None, model_name="representation_model.pth", dir="checkpoints" # 
    ):
        """
        Load the model from a directory (defaults to checkpoints).
        """
        try:
            if not os.path.exists(dir):
                if dir == "checkpoints" and os.path.exists("models"):
                    print(f"Directory '{dir}' not found, checking legacy 'models/' directory.")
                    dir = "models"
                else:
                    raise FileNotFoundError(f"Model load directory '{dir}' not found.")

            load_subdir = None
            if iteration is None or iteration == "":
                numeric_subdirs = [s for s in os.listdir(dir) if s.split('_')[0].isdigit()]
                if not numeric_subdirs:
                    raise FileNotFoundError(f"No checkpoint subdirectories found in '{dir}'.")
                latest_subdir = max(numeric_subdirs, key=lambda s: int(s.split('_')[0]))
                load_subdir = latest_subdir
                print(f"No specific iteration provided, loading from latest identified subdir: {load_subdir}")
            else:
                load_subdir = str(iteration)
                print(f"Attempting to load from specified iteration subdir: {load_subdir}")

            model_path = os.path.join(dir, load_subdir, model_name)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} not found.")

            if self.network is None and hasattr(self, 'layer_configs') and self.layer_configs:
                print(f"Rebuilding the {type(self).__name__} network before loading weights...")
                pass 

            print(f"Loading state dict from {model_path} onto device {self.device}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.load_state_dict(state_dict) 
            self.to(self.device)
            self.eval() 
            print(f"Model loaded successfully from {model_path}")

        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            raise e 
        except Exception as e:
            print(f"An unexpected error occurred during model loading: {e}")
            import traceback
            traceback.print_exc()
            raise e


    def preprocess(self, x):
        return x

    def forward(self, x):
        return self.network(x)

    def postprocess(self, x):
        return x

    def get_parameters(self):
        return list(self.network.parameters())


class RepresentationNetwork(NeuralNetwork):
     def __init__(self, layer_configs, device, build=True):
         super().__init__(layer_configs, device, build)
         self.to(device)

     def __call__(self, x, **kwargs):
         hidden_activation = self.forward(x)
         return hidden_activation

     def save_model(
         self, subdir: int | None = None, model_name="representation_model.pth", dir="models"
     ):
         super().save_model(subdir, model_name, dir)

     def load_model(
           self, iteration=None, model_name="representation_model.pth", dir="models"
       ):
           super().load_model(iteration, model_name, dir)


class DynamicsNetwork(NeuralNetwork):
    def __init__(self, layer_configs, device, action_space=18, build=True):
        head_layers = layer_configs[-4:]
        build_layers = layer_configs[:-4]

        super().__init__(build_layers, device, build)
        
        self.state_head = build_head(head_layers[0])
        self.reward_head = build_head(head_layers[1:])
        self.num_actions = action_space
        self.to(device)

    def __call__(self, hidden_state, action=None):
        if action is not None:
            # Here you may adapt the fusion scheme.
            # Hidden state is (B, C, H, W) and action is (B)
            # then concatenate along channel dimension:

            action = action.to(hidden_state.device)
            
            B, C, H, W = hidden_state.shape

            action_onehot = torch.nn.functional.one_hot(action, num_classes=self.num_actions)  # (B, num_actions)

            action_onehot = action_onehot.view(B, self.num_actions, 1, 1).float()  # (B, num_actions, 1, 1)

            action_tensor = action_onehot.expand(-1, -1, H, W)  # (B, num_actions, H, W)

            action_tensor = action_tensor.to(hidden_state.device)

            x = torch.cat((hidden_state, action_tensor), dim=1)
        else:
            x = hidden_state
            

        hidden_activation = self.forward(x)

        return self.postprocess(hidden_activation)

    def save_model(
        self, subdir: int | None = None, model_name="dynamics_model.pth", dir="models"
    ):
        super().save_model(subdir, model_name, dir)

    def preprocess(self, x, **kwargs):
        return x

    def postprocess(self, hidden_activation):
        # Return new hidden state and reward
        return self.state_head(hidden_activation), self.reward_head(hidden_activation)

    def get_parameters(self):
        return (
            super().get_parameters()
            + list(self.state_head.parameters())
            + list(self.reward_head.parameters())
        )


class PredictionNetwork(NeuralNetwork):
    def __init__(self, layer_configs, device, build=True):
        head_layers = layer_configs[-2:]
        build_layers = layer_configs[:-2]

        super().__init__(build_layers, device, build)
        
        self.policy_head = build_head(head_layers[0])
        self.value_head = build_head(head_layers[1])
        self.to(device)

    def __call__(self, x):
        input = self.preprocess(x)
        hidden_activation = self.forward(input)
        return self.postprocess(hidden_activation)

    def save_model(
        self, subdir: int | None = None, model_name="prediction_model.pth", dir="models"
    ):
        super().save_model(subdir, model_name, dir)

    def load_model(
          self, iteration=None, model_name="prediction_model.pth", dir="models"
      ):
          super().load_model(iteration, model_name, dir)

    def postprocess(self, hidden_activation):
        return self.policy_head(hidden_activation), self.value_head(hidden_activation)

    def get_parameters(self):
        return (
            super().get_parameters()
            + list(self.value_head.parameters())
            + list(self.policy_head.parameters())
        )

    def forward(self, x):
        return self.network(x)


