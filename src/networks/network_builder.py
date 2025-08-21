import torch.nn as nn
import torch

def _parse_activation(activation: str):
    if activation == "relu":
        return nn.ReLU()
    if activation == "sigmoid":
        return nn.Sigmoid()
    if activation == "tanh":
        return nn.Tanh()
    if activation == "softmax":
        return nn.Softmax(dim=1)
    raise ValueError(f"Unknown activation function: {activation}")

class ResidualBlock(nn.Module):
    """
    A basic residual block that applies two consecutive convolutional layers,
    each followed by an activation, and adds a skip connection.
    The block can be repeated multiple times.
    """
    def __init__(self, channels, kernel_size, activation, repeats=1):
        super().__init__()
        self.repeats = repeats
        self.blocks = nn.ModuleList()
        for _ in range(repeats):
            block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2),
                _parse_activation(activation),
                nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
            )
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            residual = x
            out = block(x)
            x = out + residual
        return x

class ConcatLayer(nn.Module):
    """
    A concatenation layer that concatenates inputs along a given dimension.
    The 'sources' field is provided in the config for documentation purposes.
    """
    def __init__(self, dim=1, sources=None):
        super().__init__()
        self.dim = dim
        self.sources = sources

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)

class NetworkBuilder:
    def __init__(self, layer_configs: list):
        self.layer_configs = layer_configs

    def build_network(self):
        """
        Build a network from layer configurations.
        """
        layers = []
        for layer_config in self.layer_configs:
            built_layers = self.build_layer(layer_config)
            # built_layers is a list of modules, so extend the main list.
            layers.extend(built_layers)
        return nn.Sequential(*layers)

    def build_layer(self, layer_config: dict):
        """
        Build a layer from a configuration.
        """
        layer_type = layer_config["type"]
        
        layers = []
        # For some types the 'activation' is built-in to the layer itself (e.g., residual_block)
        if layer_type == "linear":
            layers.append(self.__build_linear(layer_config))
            # Apply activation if provided.
            if "activation" in layer_config:
                layers.append(_parse_activation(layer_config["activation"]))
        elif layer_type == "conv2d":
            layers.append(self.__build_conv2d(layer_config))
            if "activation" in layer_config:
                layers.append(_parse_activation(layer_config["activation"]))
        elif layer_type == "residual_block":
            layers.append(self.__build_residual_block(layer_config))
        elif layer_type == "avg_pool2d":
            layers.append(self.__build_avg_pool2d(layer_config))
        elif layer_type == "flatten":
            layers.append(nn.Flatten())
        elif layer_type == "concat":
            # For concatenation, we assume the module will be called with multiple inputs.
            dim = layer_config.get("dim", 1)
            sources = layer_config.get("sources", None)
            layers.append(ConcatLayer(dim=dim, sources=sources))
        elif layer_type == "identity_head":
            layers.append(nn.Identity())
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        return layers

    def __build_linear(self, layer_config: dict):
        return nn.Linear(layer_config["in_features"], layer_config["out_features"])

    def __build_conv2d(self, layer_config: dict):
        return nn.Conv2d(
            layer_config["in_channels"],
            layer_config["out_channels"],
            kernel_size=layer_config["kernel_size"],
            stride=layer_config.get("stride", 1),
            padding=layer_config.get("padding", 0)
        )

    def __build_residual_block(self, layer_config: dict):
        channels = layer_config["channels"]
        kernel_size = layer_config["kernel_size"]
        repeats = layer_config.get("repeats", 1)
        activation = layer_config.get("activation", "relu")
        return ResidualBlock(channels, kernel_size, activation, repeats)

    def __build_avg_pool2d(self, layer_config: dict):
        kernel_size = layer_config["kernel_size"]
        stride = layer_config.get("stride", kernel_size)
        return nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

