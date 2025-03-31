import yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass
class LoggingConfig:
    use_wandb: bool
    wandb_project: str
    wandb_name: str
    load_model: bool
    save_model: bool
    checkpoint_interval: int

@dataclass
class NetworkConfig:
    iteration: str
    state_window: int
    hidden_state_size: int
    representation: list
    dynamics: list
    prediction: list


@dataclass
class uMCTSConfig:
    num_searches: int
    max_depth: int
    ucb_constant: float
    discount_factor: float


@dataclass
class SnakePacConfig:
    num_episodes: int
    num_episode_step: int
    training_interval: int
    buffer_size: int
    batch_size: int
    world_length: int
    seed: int
    action_space: int
    uMCTS: uMCTSConfig
    network: NetworkConfig


class Config:
    def __init__(self, config_file: str = "config.yaml"):
        self.__config_data = self.__load_yaml(config_file)

        self.environment_name = self.__config_data["environment"]
        self.logging = LoggingConfig(**self.__config_data["logging_config"])

        if self.environment_name == "snakepac":
            self.environment = SnakePacConfig(**self.__config_data["snakepac"])
            self.uMCTS = uMCTSConfig(**self.__config_data["snakepac"]["uMCTS"])
            self.networks = NetworkConfig(**self.__config_data["snakepac"]["network"])
        else:
            raise ValueError(
                f"Invalid environment: {self.environment}"
            )  # Change when new environments are added

    @staticmethod
    def __load_yaml(config_file: str):
        """Load the YAML configuration file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
        with open(config_path, "r") as file:
            return yaml.safe_load(file)


# Example usage:
if __name__ == "__main__":
    config = Config()

    print("Environment Configuration:", config.environment)
    print("Training Configuration:", config.training)
