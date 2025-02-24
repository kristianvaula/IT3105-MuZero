import yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass
class TrainingConfig:
  environment: str
  logging: bool

@dataclass
class SnakePacConfig:
  world_length: int
  seed: int

class Config:
  def __init__(self, config_file: str = "config.yaml"):
    self.config_data = self.__load_yaml(config_file)
    
    self.training = TrainingConfig(**self.config_data["training"])
    if self.training.environment == "snakepac":
      self.environment = SnakePacConfig(**self.config_data["snakepac"])
    else: 
      raise ValueError(f"Invalid environment: {self.training.environment}") # Change when new environments are added
    
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