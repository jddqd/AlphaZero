# ♟️ AlphaOne: A From-Scratch AlphaZero Implementation


## 1. Introduction

**AlphaOne** is a from-scratch implementation of AlphaZero in Python, compatible with OpenSpiel, a collection of game environments developed by Google DeepMind.

This implementation is based on the official AlphaZero approach and uses **ResNet** to estimate both the policy and the value function.

<div align="center">
  <img src="./img/resnet.png" width="100%" alt="LeCarnet Logo" />
</div>


At the core of [AlphaZero](https://arxiv.org/pdf/1712.01815) lies **Monte Carlo Tree Search** (MCTS), a heuristic search algorithm used for decision making. In this implementation, MCTS explores the most promising moves to generate training data for the ResNet, helping it refine the policy over time.

### 🎮 Supported Games

This repository currently includes full support for:

- ✅ **Tic-Tac-Toe**
- ✅ **Chess**


## 2. Quick Setup 

### OpenSpiel on Windows

Since OpenSpiel is only compatible with **Linux** and **macOS**, it is recommended that **Windows users install WSL2 (Windows Subsystem for Linux 2)** to ensure proper functionality.  
You can find more detailed instructions in the official documentation:  
[OpenSpiel Installation Guide](https://github.com/google-deepmind/open_spiel/blob/master/docs/install.md)

### Python Dependencies

All required Python packages are listed in the provided `requirements.txt` file.

```bash
# Install packages
pip install -r requirements.txt
```

### Train

_You can train a model by simply use :_

```bash
python3 train.py --game_name=chess 
```

### Play

Once the model is trained and stored in _save/_, you can test it with an interface


## 3. Adding Support for New Games

To use this algorithm with other OpenSpiel games, you need to create a new class similar to those found in the _games/_ directory. Further explanations are provided in the file _games_.






## 4. References

- [`Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm`](https://arxiv.org/pdf/1712.01815)
- [`OpenSpiel: A Framework for Reinforcement Learning in Games`](https://arxiv.org/abs/1908.09453)
- [`Youtube : AlphaZero from Scratch by Robert Förster`](https://www.youtube.com/watch?v=wuSQpLinRB4&ab_channel=freeCodeCamp.org)