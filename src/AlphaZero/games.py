import numpy as np

class Chess:
    """ Chess game representation for AlphaZero training. """

    def __init__(self):
        self.name = "chess"
        self.column_count = 8
        self.row_count = 8
        self.action_size = 4674
        self.encoded_shape = 20
        self.size_encoded = 20 * 8 * 8
    
    def normalize_observation(self, state):

        obs_tensor = np.array(state.observation_tensor()).reshape(20, 8, 8)
        current_player = state.current_player()

        tmp = obs_tensor.copy()

        if current_player == 0:
            # 180° rotation for the first 12 piece channels (even = white, odd = black)
            for i in range(0, 12, 2):
                obs_tensor[i], obs_tensor[i + 1] = (
                    np.rot90(tmp[i + 1], k=2),
                    np.rot90(tmp[i], k=2),
                )

            # Horizontal flip for the left/right channels (16 to 19)
            obs_tensor[16], obs_tensor[18] = tmp[18], tmp[16]
            obs_tensor[17], obs_tensor[19] = tmp[19], tmp[17]

        obs_tensor = obs_tensor.flatten()

        return obs_tensor.reshape(self.encoded_shape, self.column_count, self.row_count)

class TicTacToe:
    """ Tic Tac Toe game representation for AlphaZero training. """
    
    def __init__(self):
        self.name = "tic_tac_toe"
        self.column_count = 3
        self.row_count = 3
        self.action_size = 9
        self.encoded_shape = 3
        self.size_encoded = 3

    def normalize_observation(self, state):
        obs = np.array(state.observation_tensor())
        current_player = state.current_player()

        # Split 3 canals
        turn_indicator = obs[:9]
        p1_board = obs[9:18]
        p2_board = obs[18:]

        # Rearrange
        if current_player == 1:
            p1_board, p2_board = p2_board, p1_board

        normalized_obs = np.concatenate([p1_board, turn_indicator, p2_board])
        return normalized_obs.reshape(self.encoded_shape, self.column_count, self.row_count)
