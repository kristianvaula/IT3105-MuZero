import torch
import torch.nn as nn
import yaml

class ConfigurableNetwork(nn.Module):
    def __init__(self, layer_configs):
        """
        Build a network from layer configurations.
        Each configuration is a dict that includes:
          - "type": type of the layer ("linear", "conv2d", etc.)
          - Required parameters (e.g., in_features, out_features).
          - Optional "activation": Activation function name (e.g., "relu").
        """
        super(ConfigurableNetwork, self).__init__()
        layers = []
        for layer in layer_configs:
            layer_type = layer.get("type", "linear")
            if layer_type == "linear":
                layers.append(nn.Linear(layer["in_features"], layer["out_features"]))
            elif layer_type == "conv2d":
                layers.append(nn.Conv2d(
                    layer["in_channels"],
                    layer["out_channels"],
                    kernel_size=layer["kernel_size"],
                    stride=layer.get("stride", 1),
                    padding=layer.get("padding", 0)
                ))
            activation = layer.get("activation", None)
            if activation:
                if activation.lower() == "relu":
                    layers.append(nn.ReLU())
                elif activation.lower() == "tanh":
                    layers.append(nn.Tanh())
                elif activation.lower() == "sigmoid":
                    layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MuZeroWrapper(nn.Module):
    def __init__(self, config):
        """
        Constructs three networks for MuZero:
          - A representation network encoding the observation.
          - A prediction network outputting policy/value estimates.
          - A dynamics network transitioning latent states given an action.
        The network and data are automatically moved to the desired device.
        """
        super(MuZeroWrapper, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.representation_net = ConfigurableNetwork(config["representation"])
        self.dynamics_net = ConfigurableNetwork(config["dynamics"])
        self.prediction_net = ConfigurableNetwork(config["prediction"])
        self.to(self.device)

    def forward(self, observation, action=None):
        observation = observation.to(self.device)
        latent_state = self.representation_net(observation)
        prediction = self.prediction_net(latent_state)

        dynamics_output = None
        if action is not None:
            action = action.to(self.device)
            # Concatenate latent state and action for dynamics network input.
            cat_input = torch.cat([latent_state, action], dim=-1)
            dynamics_output = self.dynamics_net(cat_input)
        return latent_state, prediction, dynamics_output

if __name__ == "__main__":
    # Load the configuration from the YAML file.
    config_filename = "config.yaml"
    with open(config_filename, "r") as file:
        loaded_config = yaml.safe_load(file)

    # Set up device (uses CUDA if available).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the MuZero wrapper with the loaded configuration.
    network = MuZeroWrapper(loaded_config["network"])

    # Dummy inputs:
    observation = torch.randn(8, 64, device=device)  # Batch size 8, feature size 64.
    action = torch.randn(8, 4, device=device)         # Batch size 8, action dimension 4.

    # Run a forward pass.
    latent_state, prediction, dynamics_output = network(observation, action)
    print("Using device:", network.device)
    print("Latent state shape:", latent_state.shape)
    print("Prediction output shape:", prediction.shape)
    if dynamics_output is not None:
        print("Dynamics output shape:", dynamics_output.shape)
