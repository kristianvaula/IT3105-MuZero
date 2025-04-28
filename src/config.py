import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Union

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
    representation: List[dict]
    prediction: List[dict]
    dynamics: List[dict]
    # Optional for environments like riverraid
    state_window: Optional[int] = None
    roll_ahead: Optional[int] = None
    hidden_state_size: Optional[int] = None


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
    minibatch_size: int
    world_length: int
    seed: int
    action_space: int
    mcts: uMCTSConfig
    network: NetworkConfig


@dataclass
class RiverraidConfig:
    seed: int
    num_episodes: int
    num_episode_step: int
    training_interval: int
    buffer_size: int
    minibatch_size: int
    skip_frames: int
    action_frames: int
    mcts: uMCTSConfig
    network: NetworkConfig

class Config:
    def __init__(self, config_file: str = "config.yaml"):
        self.__config_data = self.__load_yaml(config_file)

        self.environment_name = self.__config_data["environment"]
        self.logging = LoggingConfig(**self.__config_data["logging_config"])

        if self.environment_name == "snakepac":
            env_data = self.__config_data["snakepac"]
            self.environment = SnakePacConfig(**env_data)
            self.networks =  NetworkConfig(**env_data["network"])
            self.uMCTS = uMCTSConfig(**env_data["uMCTS"])
        elif self.environment_name == "riverraid":
            env_data = self.__config_data["riverraid"]
            self.environment = RiverraidConfig(**env_data)
            self.networks =  NetworkConfig(**env_data["network"])
            self.uMCTS = uMCTSConfig(**env_data["mcts"])
        else:
            raise ValueError(f"Invalid environment: {self.environment_name}")

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
    print("Environment:", config.environment_name)
    print("Config:", config.environment)
