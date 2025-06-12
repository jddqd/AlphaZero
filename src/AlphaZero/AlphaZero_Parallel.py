from AlphaZero.MCTS import MCTSParallel
import numpy as np
import pyspiel
import torch
import torch.nn.functional as F
import random
from tqdm import trange

class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args, game_name):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)
        self.game_name = game_name
        
    def selfPlay(self):
        return_memory = []
        player = 0

        game = pyspiel.load_game(self.game_name)
        spGames = [SPG(game) for spg in range(self.args['num_parallel_games'])]
        

        while len(spGames) > 0:
            neutral_states = np.stack([spg.state.clone() for spg in spGames])
            self.mcts.search(neutral_states, spGames)
            
            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                
                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                # temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                action = np.random.choice(self.game.action_size, p=action_probs) # Divide temperature_action_probs with its sum in case of an error

                spg.state.apply_action(action)

                is_terminal = spg.state.is_terminal()
                value = spg.state.rewards()[1 - (spg.state.current_player() % 2)]
                value = np.abs(value)

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else -value
                        
                        reshaped_encoded = self.game.normalize_observation(hist_neutral_state)

                        return_memory.append((
                            reshaped_encoded,
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]
                    
            player = 1 - player
            
        return return_memory
                
    def train(self, memory):
        random.shuffle(memory)
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
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                memory += self.selfPlay()
                
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)
            
            torch.save(self.model.state_dict(), f"src/save/model_{iteration}_{self.game_name}.pt")
            torch.save(self.optimizer.state_dict(), f"src/save/optimizer_{iteration}_{self.game_name}.pt")
            
class SPG:
    def __init__(self, game):
        self.state = game.new_initial_state()
        self.memory = []
        self.root = None
        self.node = None