# MuZero Knockoff – IT3105 Spring 2025  

**Course:** AI Programming (IT-3105) – NTNU  
**Due Date:** May 2, 2025  

**Team Members**  
- Aleksander Olsvik  
- Edvard Schøyen  
- Kristian Vaula Jensen  
- Vetle Ekern  

---

## 📖 Project Overview  
This project is our implementation of a **MuZero-inspired reinforcement learning system**, developed as the main project for *AI Programming (IT3105)*. MuZero, introduced by DeepMind in 2019, combines model-based and model-free reinforcement learning by simultaneously learning:  

1. **A representation network (NNr)** – maps real game states to abstract latent states.  
2. **A dynamics network (NNd)** – predicts the next latent state and immediate reward given a latent state and action.  
3. **A prediction network (NNp)** – outputs a policy distribution and value estimate from a latent state.  

Using these three components, MuZero builds a search tree with **Monte Carlo Tree Search (u-MCTS)**, guiding the agent’s decisions while also generating training targets.  

We re-implemented the **core pipeline of MuZero from scratch**, including training with backpropagation through time (BPTT), episodic buffers, and integration with custom environments.  

---

## 🧩 Project Structure  

```
src/
├── config/ # Configuration handling (YAML)
├── envs/ # Game environments (SnakePac, RiverRaid, wrappers)
├── gsm/ # Game State Managers
├── networks/ # Representation, Dynamics, Prediction networks
├── self_play/ # u-MCTS implementation
├── storage/ # Episode buffer for replay and training
├── rlm.py # Reinforcement Learning Manager (main training loop)
tests/ # Unit tests
models/ # Saved models/checkpoints
episode_data/ # Training episode storage
```
Key modules:  
- **`ReinforcementLearningManager`** – orchestrates training and evaluation.  
- **`uMCTS`** – abstract-state Monte Carlo Tree Search implementation.  
- **`NeuralNetManager`** – coordinates training of the three MuZero networks.  
- **`SnakePacEnv` / `RiverraidEnv`** – custom RL environments used for testing.  
- **`EpisodeBuffer`** – stores episodic data for BPTT training.  

---

## 🎮 Environments  
We implemented and tested MuZero in two environments:  

- **SnakePac (custom)**  
  - A simplified grid-world with coins and movement actions.  
  - Designed for debugging and rapid iteration.  

- **River Raid (Atari)**  
  - Leveraged ALE via Gymnasium.  
  - High-dimensional pixel input tested the scalability of our MuZero implementation.  

---

## ⚙️ Training Pipeline  

1. **Episode Generation**  
   - The agent plays episodes using **u-MCTS** to select actions.  
   - Each step records:  
     - State  
     - Value estimate  
     - Policy distribution  
     - Action taken  
     - Reward received  

2. **Episode Buffer**  
   - Stores episode histories.  
   - Supports sampling windows of states and actions for training.  

3. **Backpropagation Through Time (BPTT)**  
   - Trains all three networks jointly.  
   - Uses a sliding window of past and future states (`q` look-back, `w` roll-ahead).  

4. **Checkpointing & Logging**  
   - Checkpoints saved under `checkpoints/`.  
   - Optional logging with **Weights & Biases (wandb)** for monitoring.  

---

## 🔑 Key Learnings

Implementing MuZero required integrating planning and learning tightly, reinforcing the idea of extreme bootstrapping.

Working with Atari-scale environments highlighted the challenges of training efficiency and compute limitations.

Producing educational visualizations was just as important as the code, sharpening our ability to communicate complex AI concepts clearly.
