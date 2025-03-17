from src.config import Config
from src.storage.episode_buffer import EpisodeBuffer
from src.self_play.uMCTS import uMCTS
from src.networks.network_builder import NeuralNetwork
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
        build = config.networks.iteration is None
        if build:
            # TODO: Add method for calculating input layer
            representation_network = NeuralNetwork(config.networks.representation, device="cuda", build=build)
            dynamics_network = NeuralNetwork(config.networks.dynamics, device="cuda", build=build)
            prediction_network = NeuralNetwork(config.networks.prediction, device="cuda", build=build)
        
        else:
            representation_network = NeuralNetwork(config.networks.representation, device="cuda", build=build)
            dynamics_network = NeuralNetwork(config.networks.dynamics, device="cuda", build=build)
            prediction_network = NeuralNetwork(config.networks.prediction, device="cuda", build=build)
            representation_network.load_model(config.networks.iteration, "representation_model.pth")
            dynamics_network.load_model(config.networks.iteration, "dynamics_model.pth")
            prediction_network.load_model(config.networks.iteration, "prediction_model.pth")
            
        # Initialize neural network manager (NNM)
        nnm = NeuralNetManager(representation_network, dynamics_network, prediction_network)

        # Intialize u-MCTS module
        monte_carlo = uMCTS(nnm, gsm, env.action_space, config.uMCTS.num_searches,
            config.uMCTS.max_depth, config.uMCTS.ucb_constant, config.uMCTS.discount_factor)

        # Initialize episode buffer
        episode_buffer = EpisodeBuffer()

        # Initalize reinforcement learning manager (RLM)
        self.rlm = ReinforcementLearningManager(env, gsm, monte_carlo, nnm, episode_buffer, config)
        # return monte_carlo, episode_buffer

    def run_training(self):
        pass  # self.rlm.train();

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


if __name__ == "__main__":
    main()
