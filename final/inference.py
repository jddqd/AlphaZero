import matplotlib.pyplot as plt
import pyspiel  # OpenSpiel
import torch
from games import TicTacToe
from ResNet import ResNet
from configs import AlphaConfig
from dataclasses import asdict

game = TicTacToe()

config = AlphaConfig()
args = asdict(config)

openspiel = pyspiel.load_game("tic_tac_toe")
state = openspiel.new_initial_state()

actions = [0,1,3,2]
for action in actions:
    state.apply_action(action)


model = ResNet(game, 4, 64, device=args['device'])
model.load_state_dict(torch.load('final/save/model_2.pt'))
model.eval()

encoded_state = game.normalize_observation(state)
reshaped_encoded = encoded_state.reshape(game.encoded_shape, game.column_count, game.row_count)
tensor_state = torch.tensor(reshaped_encoded, dtype=torch.float32, device=model.device).unsqueeze(0)

# Predictions
policy, value = model(tensor_state)
value = value.item()
policy = torch.softmax(policy, dim=1).squeeze(0).detach().cpu().numpy()

print("Value prediction:", value)
print("Current state:\n", state)
print("Tensor state shape:", tensor_state.shape)

plt.bar(range(openspiel.num_distinct_actions()), policy)
plt.xlabel("Actions")
plt.ylabel("Probability")
plt.title("Predicted Policy")
plt.show()