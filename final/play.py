import numpy as np
from games import TicTacToe
import pyspiel
from MCTS import MCTS
from ResNet import ResNet
import torch
from configs import AlphaConfig
from dataclasses import asdict

# Initialize the TicTacToe game
tictactoe = pyspiel.load_game("tic_tac_toe")
game = TicTacToe()

state = tictactoe.new_initial_state()

player = 1

config = AlphaConfig()
args = asdict(config)

model = ResNet(game, 4, 64, device=args['device'])
model.load_state_dict(torch.load('final/save/model_2.pt'))
model.eval()

mcts = MCTS(game, args, model)



while True:
    print(state)
    
    if player == 1:
        valid_moves = state.legal_actions_mask()
        print("valid_moves", state.legal_actions())
        action = int(input(f"{player}:"))

        if valid_moves[action] == 0:
            print("action not valid")
            continue
            
    else:
        mcts_probs = mcts.search(state.clone())
        action = np.argmax(mcts_probs)
        
    state.apply_action(action)
    
    is_terminal = state.is_terminal()
    value = state.rewards()[state.current_player() % 2]

    print("value", value)
    
    if is_terminal:
        print(state)
        if value == 0.0:
            print("draw")
        else:
            print(player, "won")
        break

    player = 1 - player