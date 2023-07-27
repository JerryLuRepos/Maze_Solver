import datetime, json, random
import numpy as np
from collections import deque
from Create_Maze import create_maze
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizer_v2.adam import Adam
from keras.layers.advanced_activations import PReLU

# LEFT = 0
# UP = 1
# RIGHT = 2
# DOWN = 3

class deepq():
    def __init__(self, maze):
        # Initialize attributes
        # "n_episodes": number of full single path
        # "max_exp_len": number of stored experience
        # "batch_size": number of samples pulled from experience for retraining
        # "n_epochs": number of retraining times
        self.configs = json.load(open('config.json', 'r'))['deep_q']
        self.maze = maze
        self.state_size = np.array(maze).size
        self.nrow = len(maze)
        self.ncol = len(maze[0])
        self.default_state = self.maze_conversion()
        self.target = tuple([[row, col] for row in range(self.nrow) for col in range(self.ncol) if self.maze[row, col] == 'G'][0])
        self.free_cells = [(r, c) for r in range(self.nrow) for c in range(self.ncol) if self.default_state[r, c] == 1]
        self.free_cells.remove(self.target)
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()

        self.action_size = self.configs['action_size']
        self.optimizer = Adam(learning_rate=self.configs['learning_rate'])
        self.experience = deque([], maxlen=self.configs['max_exp_len'])

        # Build networks
        self.q_network = self.build_model()
        self.q_shortest_path = []

    def build_model(self):
        # construct model
        model = Sequential()
        model.add(Dense(self.state_size, input_shape=(self.state_size,)))
        model.add(PReLU())
        model.add(Dense(self.state_size))
        model.add(PReLU())
        model.add(Dense(self.action_size))
        model.compile(optimizer=self.optimizer, loss='mse')
        return model

    def maze_conversion(self):
        """Converts maze into rewards table
        :return: numpy array of rewards
        """
        default_state = []
        for i in range(self.nrow):
            row = []
            for j in range(self.ncol):
                cur = self.maze[i][j]
                if cur in ['w']:
                    row.append(0)
                else:
                    row.append(1)
            default_state.append(np.array(row))
        return np.array(default_state)

    def observe_state(self, cur_cell):
        # get current state with current cell being 0.5
        cur_row, cur_col = cur_cell[0], cur_cell[1]
        state = self.default_state.copy()
        state[cur_row, cur_col] = 0.5
        return state.reshape((1, -1))

    def valid_actions(self, cell):
        # select actions that don't hit walls or outside of maze
        row, col = cell
        actions = list(range(0, self.action_size))

        if self.default_state[row - 1, col] == 0:
            actions.remove(1)
        if self.default_state[row + 1, col] == 0:
            actions.remove(3)
        if self.default_state[row, col - 1] == 0:
            actions.remove(0)
        if self.default_state[row, col + 1] == 0:
            actions.remove(2)

        return actions

    def predict(self, envstate):
        return self.q_network.predict(envstate)[0]

    def get_epsilon(self, episode):
        """Calculates Epsilon Decay
        :param episode: number of path ran
        :return: Decay Epsilon
        """
        epsilon = self.configs['epsilon'] # 0.1 -> 0.9
        decay_rate = self.configs['epsilon_decay']
        min_epsilon = self.configs['min_epsilon']
        return max(min_epsilon, epsilon*(decay_rate**episode))

    def get_action(self, valid_actions, envstate, episode):
        # get next action based on epsilon
        if np.random.rand() < self.get_epsilon(episode):
            return random.choice(valid_actions)
        else:
            return np.argmax(self.predict(envstate))

    def get_next_location(self, cur_cell, action):
        """Get location of next action
        """
        cur_row, cur_col = cur_cell
        if action == 1 and cur_row > 0:
            cur_row -= 1
        elif action == 2 and cur_col < self.ncol - 1:
            cur_col += 1
        elif action == 3 and cur_row < self.nrow - 1:
            cur_row += 1
        elif action == 0 and cur_col > 0:
            cur_col -= 1

        next_cell = (cur_row, cur_col)
        return next_cell

    def get_reward(self, next_cell):
        next_row, next_col = next_cell
        target_row, target_col = self.target
        if target_row == next_row and target_col == next_col:
            return 1.0
        if self.default_state[next_row, next_col] == 0:
            return self.min_reward - 1
        elif (next_row, next_col) in self.visited:
            return -0.25
        else:
            return -0.04

    def game_status(self, next_cell):
        next_row, next_col = next_cell
        target_row, target_col = self.target
        if self.total_reward < self.min_reward:
            return 'lose'
        elif next_row == target_row and next_col == target_col:
            return 'win'

        return 'not_over'

    def act(self, next_cell):
        reward = self.get_reward(next_cell)
        self.total_reward += reward
        status = self.game_status(next_cell)
        next_envstate = self.observe_state(next_cell)
        return next_envstate, reward, status

    def get_data(self, data_size=10):
        env_size = self.state_size
        mem_size = len(self.experience)
        data_size = min(mem_size, data_size)
        discounted_rate = self.configs['discounted_rate']
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.action_size))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.experience[j]
            inputs[i] = envstate
            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + discounted_rate * Q_sa
        return inputs, targets

    def qtrain(self):
        n_episodes = self.configs['n_episodes']
        batch_size = self.configs['batch_size']
        max_steps = self.configs['max_steps']
        start_time = datetime.datetime.now()
        model = self.q_network

        win_rate = 0.0
        win_history = []  # history of win/lose game

        for episode in range(n_episodes):
            cur_cell = random.choice(self.free_cells)
            self.visited = set()
            self.total_reward = 0
            loss = None
            game_over = False

            n_steps = 0
            while not game_over and n_steps < max_steps:
                envstate = self.observe_state(cur_cell)  # flatten 1d array
                valid_actions = self.valid_actions(cur_cell)
                if not valid_actions: break

                action = self.get_action(valid_actions, envstate, episode)
                self.visited.add(cur_cell)
                next_cell = self.get_next_location(cur_cell, action)

                # Apply action, get reward and new envstate
                next_envstate, reward, game_status = self.act(next_cell)
                if game_status == 'win':
                    win_history.append(1)
                    game_over = True
                elif game_status == 'lose':
                    win_history.append(0)
                    game_over = True
                else:
                    game_over = False

                # store experience
                self.experience.append([envstate, action, reward, next_envstate, game_over])
                n_steps += 1

                # Train neural network model
                inputs, targets = self.get_data(data_size=batch_size)
                model.fit(
                    inputs,
                    targets,
                    epochs=1,  # 8
                    batch_size=16,
                    verbose=0,
                )
                loss = model.evaluate(inputs, targets, verbose=0)
                # print('Steps: ', n_steps, 'reward: ', reward, cur_cell, next_cell, game_status, game_over)
                cur_cell = next_cell

            # check win rate based on half of maze size
            hsize = self.state_size//2
            if len(win_history) > hsize:
                win_rate = sum(win_history[-hsize:])/hsize

            minutes = round((datetime.datetime.now() - start_time).total_seconds()/60, 2)
            template = "Episode: {:03d}/{:d} | Loss: {:.4f} | Steps: {:d} | Win count: {:d} | Win rate: {:.3f} | Epsilon: {:.2f} | time: {} Minutes"
            print(template.format(episode, n_episodes - 1, loss, n_steps, sum(win_history), win_rate, self.get_epsilon(episode), minutes))
            if win_rate > 0.9: self.configs['epsilon'] = 0.05
            if win_rate == 1: break

    def check_invalid_state(self, row, col):
        """Checks if current location is invalid
        :param row: row idx
        :param col: col idx
        :return: True or False
        """
        if self.default_state[row, col] == 0:
            return False
        else:
            return True

    def get_shortest_path(self, start_row_index, start_column_index):
        """Get shortest path based on max q values
        :param start_row_index: row
        :param start_column_index: col
        :return: a list of location on shortest path
        """
        cur_row, cur_col = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([cur_row, cur_col])
        while not self.check_invalid_state(cur_row, cur_col):
            state = self.observe_state([cur_row, cur_col]).reshape(1, -1)
            action_index = np.argmax(self.q_network.predict(state)[0])
            cur_row, cur_col = self.get_next_location([cur_row, cur_col], action_index)
            if [cur_row, cur_col] not in shortest_path:
                shortest_path.append([cur_row, cur_col])
            else:
                print('Loop Detected!')
                break
        self.q_shortest_path = shortest_path


maze = np.array([
    ['w',  'w',  'w',  'w',  'w',  'w',  'w',  'w',  'w'],
    ['w',  '.',  'w',  '.',  '.',  '.',  '.',  '.',  'w'],
    ['w',  '.',  '.',  '.',  'w',  'w',  '.',  'w',  'w'],
    ['w',  'w',  'w',  'w',  '.',  '.',  '.',  'w',  'w'],
    ['w',  '.',  '.',  '.',  '.',  'w',  'w',  '.',  'w'],
    ['w',  '.',  'w',  'w',  'w',  '.',  '.',  '.',  'w'],
    ['w',  '.',  'w',  '.',  '.',  '.',  '.',  '.',  'w'],
    ['w',  '.',  '.',  '.',  'w',  '.',  '.',  'G',  'w'],
    ['w',  'w',  'w',  'w',  'w',  'w',  'w',  'w',  'w']
])

deep_q_learning = deepq(maze)
deep_q_learning.qtrain()
deep_q_learning.get_shortest_path(0, 0)
print(deep_q_learning.q_shortest_path)