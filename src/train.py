import AlphaZero.games as games
import torch
from AlphaZero.ResNet import ResNet
from configs import AlphaConfig
from dataclasses import asdict
from AlphaZero.AlphaZero_Parallel import AlphaZeroParallel
import argparse


def main(args):

    config = AlphaConfig()
    config_dict = asdict(config)

    print("Using device:", config_dict['device'])

    if args.game_name.lower() not in ["chess", "tictactoe"]:
        raise ValueError("Invalid game name. Please choose 'Chess' or 'TicTacToe'.")

    if args.game_name.lower() == "chess":
        game = games.Chess()
    elif args.game_name.lower() == "tictactoe":
        game = games.TicTacToe()


    model = ResNet(game, config_dict['num_resBlocks'], config_dict['num_hidden'], device=config_dict['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    if args.game_name.lower() == "chess":
        alphaZero = AlphaZeroParallel(model, optimizer, game, config_dict, "chess")
    else:
        alphaZero = AlphaZeroParallel(model, optimizer, game, config_dict, "tic_tac_toe")

    alphaZero.learn()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AlphaZero Training Script",
    )
    parser.add_argument(
        "--game_name",
        type=str,
        default="TicTacToe",
        help="Name of the game to train on (Chess or TicTacToe).",
    )
    args = parser.parse_args()
    main(args)
