import torch
from src.config import Config
from src.storage.episode_buffer import EpisodeBuffer
from src.self_play.uMCTS import uMCTS
from src.networks.neural_network import RepresentationNetwork, DynamicsNetwork, PredictionNetwork
from src.networks.neural_network_manager import NeuralNetManager
from src.rlm import ReinforcementLearningManager

""" Important parameters 

num_episodes: Number of episodes to run
num_simulations: Number of simulations to run in u-MCTS
ucb_constant: UCB constant for u-MCTS
mbs (Minibatch size): Number of episodes to sample for BPTT
I_t: Training interval for the trident network


"""


class MuZero:
    def __init__(self, config: Config):
        # Load environments and Game State Manager
        env, gsm = self.__initialize_env(config)

        # Initialize neural networks with configurations
        # TODO
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if config.load_model and config.networks.iteration is not None:
            representation_network = RepresentationNetwork(config.networks.representation, device=device)
            dynamics_network = DynamicsNetwork(config.networks.dynamics, device=device)
            prediction_network = PredictionNetwork(config.networks.prediction, device=device)
            representation_network.load_model(config.networks.iteration, "representation_model.pth")
            dynamics_network.load_model(config.networks.iteration, "dynamics_model.pth")
            prediction_network.load_model(config.networks.iteration, "prediction_model.pth")
        else:
            # TODO: Add method for calculating input layer
            representation_network = RepresentationNetwork(config.networks.representation, device=device)
            dynamics_network = DynamicsNetwork(config.networks.dynamics, device=device)
            prediction_network = PredictionNetwork(config.networks.prediction, device=device)
            
        # Initialize neural network manager (NNM)
        self.nnm = NeuralNetManager(representation_network, dynamics_network, prediction_network)

        # Intialize u-MCTS module
        monte_carlo = uMCTS(self.nnm, gsm, env.action_space, config.uMCTS.num_searches,
            config.uMCTS.max_depth, config.uMCTS.ucb_constant, config.uMCTS.discount_factor)

        # Initialize episode buffer
        episode_buffer = EpisodeBuffer()

        # Initalize reinforcement learning manager (RLM)
        self.rlm = ReinforcementLearningManager(env, gsm, monte_carlo, self.nnm, episode_buffer, config)

    def run_training(self):
        self.rlm.train()
        
    def save_models(self):
        self.nnm.representation.save_model()
        self.nnm.dynamics.save_model()
        self.nnm.prediction.save_model()

    def __initialize_env(self, config: Config):
        if config.environment_name == "snakepac":
            from .environments.snakepac import SnakePacEnv
            from .gsm.snakepac_gsm import SnakePacGSM

            env = SnakePacEnv(config.environment.world_length, config.environment.seed)
            gsm = SnakePacGSM(env)
        else:
            raise ValueError(f"Invalid environment: {config.environment}")

        return env, gsm


def main():
    # Load configurations
    config = Config()

    # Initialize MuZero
    muzero = MuZero(config)

    # Run training loop
    muzero.run_training()
    if config.save_model:
        muzero.save_models()
        
    muzero.rlm.play(5)


if __name__ == "__main__":
    main()
