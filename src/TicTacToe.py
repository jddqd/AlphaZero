# MODULES
import pygame, sys
import numpy as np
from games import TicTacToe
import pyspiel
from MCTS import MCTS
from ResNet import ResNet
import torch
from configs import AlphaConfig
from dataclasses import asdict

# Adapted from https://github.com/AlejoG10/python-tictactoe-yt/

# initializes pygame
pygame.init()

# ---------
# CONSTANTS
# ---------
WIDTH = 600
HEIGHT = 600
LINE_WIDTH = 15
WIN_LINE_WIDTH = 15
BOARD_ROWS = 3
BOARD_COLS = 3
SQUARE_SIZE = 200
CIRCLE_RADIUS = 60
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = 55
# rgb: red green blue
RED = (255, 0, 0)
BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)

# ------
# SCREEN
# ------
screen = pygame.display.set_mode( (WIDTH, HEIGHT) )
pygame.display.set_caption( 'TIC TAC TOE vs AlphaZero' )
screen.fill( BG_COLOR )

# -------------
# GAME SETUP
# -------------
# Initialize the TicTacToe game
tictactoe = pyspiel.load_game("tic_tac_toe")
game = TicTacToe()

config = AlphaConfig()
args = asdict(config)

model = ResNet(game, 4, 64, device=args['device'])
model.load_state_dict(torch.load('final/save/model_2.pt'))
model.eval()

mcts = MCTS(game, args, model)

# Console board for pygame display
board = np.zeros( (BOARD_ROWS, BOARD_COLS) )

# ---------
# FUNCTIONS
# ---------
def draw_lines():
	# 1 horizontal
	pygame.draw.line( screen, LINE_COLOR, (0, SQUARE_SIZE), (WIDTH, SQUARE_SIZE), LINE_WIDTH )
	# 2 horizontal
	pygame.draw.line( screen, LINE_COLOR, (0, 2 * SQUARE_SIZE), (WIDTH, 2 * SQUARE_SIZE), LINE_WIDTH )

	# 1 vertical
	pygame.draw.line( screen, LINE_COLOR, (SQUARE_SIZE, 0), (SQUARE_SIZE, HEIGHT), LINE_WIDTH )
	# 2 vertical
	pygame.draw.line( screen, LINE_COLOR, (2 * SQUARE_SIZE, 0), (2 * SQUARE_SIZE, HEIGHT), LINE_WIDTH )

def draw_figures():
	for row in range(BOARD_ROWS):
		for col in range(BOARD_COLS):
			if board[row][col] == 1:
				pygame.draw.circle( screen, CIRCLE_COLOR, (int( col * SQUARE_SIZE + SQUARE_SIZE//2 ), int( row * SQUARE_SIZE + SQUARE_SIZE//2 )), CIRCLE_RADIUS, CIRCLE_WIDTH )
			elif board[row][col] == 2:
				pygame.draw.line( screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE), (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE), CROSS_WIDTH )	
				pygame.draw.line( screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE), (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE), CROSS_WIDTH )

def mark_square(row, col, player):
	board[row][col] = player

def available_square(row, col):
	return board[row][col] == 0

def is_board_full():
	for row in range(BOARD_ROWS):
		for col in range(BOARD_COLS):
			if board[row][col] == 0:
				return False
	return True

def check_win(player):
	# vertical win check
	for col in range(BOARD_COLS):
		if board[0][col] == player and board[1][col] == player and board[2][col] == player:
			draw_vertical_winning_line(col, player)
			return True

	# horizontal win check
	for row in range(BOARD_ROWS):
		if board[row][0] == player and board[row][1] == player and board[row][2] == player:
			draw_horizontal_winning_line(row, player)
			return True

	# asc diagonal win check
	if board[2][0] == player and board[1][1] == player and board[0][2] == player:
		draw_asc_diagonal(player)
		return True

	# desc diagonal win chek
	if board[0][0] == player and board[1][1] == player and board[2][2] == player:
		draw_desc_diagonal(player)
		return True

	return False

def draw_vertical_winning_line(col, player):
	posX = col * SQUARE_SIZE + SQUARE_SIZE//2

	if player == 1:
		color = CIRCLE_COLOR
	elif player == 2:
		color = CROSS_COLOR

	pygame.draw.line( screen, color, (posX, 15), (posX, HEIGHT - 15), LINE_WIDTH )

def draw_horizontal_winning_line(row, player):
	posY = row * SQUARE_SIZE + SQUARE_SIZE//2

	if player == 1:
		color = CIRCLE_COLOR
	elif player == 2:
		color = CROSS_COLOR

	pygame.draw.line( screen, color, (15, posY), (WIDTH - 15, posY), WIN_LINE_WIDTH )

def draw_asc_diagonal(player):
	if player == 1:
		color = CIRCLE_COLOR
	elif player == 2:
		color = CROSS_COLOR

	pygame.draw.line( screen, color, (15, HEIGHT - 15), (WIDTH - 15, 15), WIN_LINE_WIDTH )

def draw_desc_diagonal(player):
	if player == 1:
		color = CIRCLE_COLOR
	elif player == 2:
		color = CROSS_COLOR

	pygame.draw.line( screen, color, (15, 15), (WIDTH - 15, HEIGHT - 15), WIN_LINE_WIDTH )

def restart():
	global state, player
	screen.fill( BG_COLOR )
	draw_lines()
	for row in range(BOARD_ROWS):
		for col in range(BOARD_COLS):
			board[row][col] = 0
	# Reset game state
	state = tictactoe.new_initial_state()
	player = 1

def convert_click_to_action(row, col):
	"""Convert pygame grid position to pyspiel action"""
	return row * 3 + col

def convert_action_to_grid(action):
	"""Convert pyspiel action to pygame grid position"""
	row = action // 3
	col = action % 3
	return row, col

def update_pygame_board_from_state(state):
	"""Update pygame board from pyspiel state"""
	global board
	board = np.zeros((BOARD_ROWS, BOARD_COLS))
	
	# Get the string representation and parse it
	state_str = str(state).strip()
	lines = state_str.split('\n')
	
	for i, line in enumerate(lines):
		if i < 3:  # Only first 3 lines contain the board
			for j, char in enumerate(line):
				if char == 'x':
					board[i][j] = 1  # Player 1 (human)
				elif char == 'o':
					board[i][j] = 2  # Player 2 (AI)

def ai_move():
	"""Make AlphaZero move using MCTS"""
	global state, player
	print("AlphaZero is thinking...")
	mcts_probs = mcts.search(state.clone())
	action = np.argmax(mcts_probs)
	
	# Apply action to game state
	state.apply_action(action)
	
	# Update pygame board
	row, col = convert_action_to_grid(action)
	mark_square(row, col, 2)  # AI is player 2
	
	print(f"AlphaZero played action {action} at position ({row}, {col})")

draw_lines()

# ---------
# VARIABLES
# ---------
state = tictactoe.new_initial_state()
player = 1  # 1 = human, 2 = AI
game_over = False

# --------
# MAINLOOP
# --------
while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			sys.exit()

		if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
			if player == 1:  # Human turn
				mouseX = event.pos[0] # x
				mouseY = event.pos[1] # y

				clicked_row = int(mouseY // SQUARE_SIZE)
				clicked_col = int(mouseX // SQUARE_SIZE)

				if available_square( clicked_row, clicked_col ):
					# Make human move
					action = convert_click_to_action(clicked_row, clicked_col)
					
					# Check if move is valid in pyspiel
					if action in state.legal_actions():
						# Apply to both boards
						mark_square( clicked_row, clicked_col, player )
						state.apply_action(action)
						
						# Check for win/draw
						if check_win( player ):
							game_over = True
							print("Human wins!")
						elif is_board_full():
							game_over = True
							print("Draw!")
						else:
							player = 2
						
						draw_figures()

		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_r:
				restart()
				player = 1
				game_over = False

	# AlphaZero turn (outside event loop to avoid blocking)
	if player == 2 and not game_over:
		if not state.is_terminal():
			ai_move()
			
			# Check for win/draw after AlphaZero move
			if check_win(2):
				game_over = True
				print("AlphaZero wins!")
			elif is_board_full():
				game_over = True
				print("Draw!")
			else:
				player = 1  # Switch back to human
			
			draw_figures()
		else:
			game_over = True
			# Check final result
			rewards = state.rewards()
			if rewards[0] == 1:  # Human wins
				print("Human wins!")
			elif rewards[1] == 1:  # AlphaZero wins
				print("AlphaZero wins!")
			else:
				print("Draw!")

	pygame.display.update()