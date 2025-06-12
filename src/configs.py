from dataclasses import dataclass
import torch


@dataclass
class AlphaConfig:
    C: int = 2
    num_searches: int = 60
    num_iterations: int = 3
    num_selfPlay_iterations: int = 500
    num_parallel_games: int = 10
    num_epochs: int = 4
    batch_size: int = 64
    num_resBlocks: int = 4
    num_hidden: int = 128
    # temperature: float = 1.0  # Uncomment and include if needed
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = 0.3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"