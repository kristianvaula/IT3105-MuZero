import torch.nn as nn

class NetworkBuilder:
    def __init__(self, layer_configs: dict):
        self.layer_configs = layer_configs
        

    def build_network(self):
        """
        Build a network from layer configurations.
        """
        layers = []
        for layer_config in self.layer_configs:
            layer = self.build_layer(layer_config)
            layers.extend(layer)
                
        return nn.Sequential(*layers)
    
    def build_layer(self, layer_config: dict):
        """
        Build a layer from a configuration.
        """
        layer_type = layer_config["type"]
        layers = []
        activation = layer_config.get("activation", None)
        if layer_type == "linear":
            layers.append(self.__build_linear(layer_config))
        elif layer_type == "conv2d":
            layers.append(self.__build_conv2d(layer_config))
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        if activation is not None:
                activation_layer = self.__parse_activation(activation)
                layers.append(activation_layer)
                
        return layers

    def __build_linear(self, layer_config):
        """
        Build a linear layer from a configuration.
        """
        return nn.Linear(layer_config["in_features"], layer_config["out_features"])

    def __build_conv2d(self, layer_config: dict):
        """
        Build a 2D convolutional layer from a configuration.
        """
        return nn.Conv2d(
            layer_config["in_channels"],
            layer_config["out_channels"],
            kernel_size=layer_config["kernel_size"],
            stride=layer_config.get("stride", 1),
            padding=layer_config.get("padding", 0)
        )

    def __parse_activation(self, activation: str):
        if activation == "relu":
            return nn.ReLU()
        if activation == "sigmoid":
            return nn.Sigmoid()
        if activation == "tanh":
            return nn.Tanh()
        if activation == "softmax":
            return nn.Softmax(dim=1)
        raise ValueError(f"Unknown activation function: {activation}")