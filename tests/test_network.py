import os
import time
import shutil
import torch
import pytest
from src.networks.network_builder import NetworkBuilder
from src.networks.neural_network import NeuralNetwork

# Sample network configuration for testing
layer_configs = [
    {"type": "linear", "in_features": 64, "out_features": 128, "activation": "relu"},
    {"type": "linear", "in_features": 128, "out_features": 64}
]

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper function to clean up saved model directories
def cleanup_models():
    if os.path.exists("models"):
        shutil.rmtree("models")

@pytest.fixture(scope="module")
def neural_network():
    """Creates a NeuralNetwork instance for testing."""
    return NeuralNetwork(layer_configs, device=device)

# Test that NetworkBuilder constructs a valid model
def test_network_builder():
    builder = NetworkBuilder(layer_configs)
    model = builder.build_network()

    assert isinstance(model, torch.nn.Sequential), "Network is not an instance of nn.Sequential"
    assert len(model) == 3, f"Expected 3 layers, got {len(model)}"

    sample_input = torch.randn(1, 64)
    output = model(sample_input)

    assert output.shape == (1, 64), f"Incorrect output shape: {output.shape}"

# Test initialization of NeuralNetwork
def test_neural_network_init(neural_network):
    assert neural_network.network is not None, "NeuralNetwork instance failed to initialize"

# Test saving a model
def test_save_model(neural_network):
    timestamp = int(time.time())
    neural_network.save_model(timestamp)

    saved_path = f"models/{timestamp}/representation_model.pth"
    assert os.path.exists(saved_path), f"Model was not saved at {saved_path}"

# Test loading a saved model and verifying that weights are preserved
def test_load_model(neural_network):
    timestamp = int(time.time())
    neural_network.save_model(timestamp)

    new_nn_model = NeuralNetwork(layer_configs, device=device, build=False)
    new_nn_model.load_model(timestamp)

    for param1, param2 in zip(neural_network.network.parameters(), new_nn_model.network.parameters()):
        assert torch.equal(param1, param2), "Weights do not match after loading"

# Test forward pass with valid input data
def test_forward_pass(neural_network):
    sample_input = torch.randn(1, 64).to(device)
    output = neural_network.forward(sample_input)

    assert output.shape == (1, 64), f"Forward pass output shape is incorrect: {output.shape}"

# Test handling of a missing model checkpoint
def test_load_nonexistent_model():
    nn_model = NeuralNetwork(layer_configs, device=device, build=False)
    nn_model.load_model(iteration=99999999)  # Attempting to load a non-existent checkpoint

# Test that saving fails when the model has not been initialized
def test_save_uninitialized_model():
    try:
        nn_model = NeuralNetwork(layer_configs, device=device, build=False)
        nn_model.save_model(int(time.time()))
        assert False, "Expected an error when saving an uninitialized model"
    except ValueError as e:
        pass  # Expected behavior

# Test that the model correctly rebuilds before loading weights
def test_model_rebuild_on_load():
    timestamp = int(time.time())
    nn_model = NeuralNetwork(layer_configs, device=device)
    nn_model.save_model(timestamp)

    new_nn_model = NeuralNetwork(layer_configs, device=device, build=False)
    new_nn_model.load_model(timestamp)

    assert new_nn_model.network is not None, "Network was not rebuilt before loading weights"

# Test that temporary model directories are properly cleaned up
def test_cleanup():
    cleanup_models()
    assert not os.path.exists("models"), "Cleanup failed"

if __name__ == "__main__":
    cleanup_models()
    pytest.main()
