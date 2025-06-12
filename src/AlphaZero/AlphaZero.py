
from AlphaZero.MCTS import MCTS
import numpy as np
import pyspiel
import torch
import torch.nn.functional as F
import random
from tqdm import trange



class AlphaZero:
    """ AlphaZero class for implementing the AlphaZero algorithm."""
    
    def __init__(self, model, optimizer, game, args, game_name):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        self.game_name = game_name
        
    def selfPlay(self):
        """ Self-play method to generate training data."""

        memory = []
        player = 0

        pyspiel_game = pyspiel.load_game(self.game_name)
        state = pyspiel_game.new_initial_state()
        
        while True:
            neutral_state = state.clone()
            action_probs = self.mcts.search(neutral_state)
            memory.append((neutral_state, action_probs, player))
            
            action = np.random.choice(self.game.action_size, p=action_probs)
            state.apply_action(action)
            
            is_terminal = state.is_terminal()
            value = state.rewards()[1 - (state.current_player() % 2)]
            value = np.abs(value)

            
            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else -value
                    
                    encoded_state = self.game.normalize_observation(hist_neutral_state)

                    returnMemory.append((
                        encoded_state,
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory
            
            player = 1 - player


    
    def train(self, memory):
        """ Train the model using the generated memory."""

        random.shuffle(memory)
        # train batch per batch
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets, value_targets = zip(*sample)
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    
    def learn(self):
        """ Main learning loop for the AlphaZero algorithm."""

        print("Starting AlphaZero learning process...")

        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()
                
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)
            
            torch.save(self.model.state_dict(), f"final/save/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"final/save/optimizer_{iteration}.pt")
