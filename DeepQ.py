import datetime, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.advanced_activations import PReLU


class Qmaze(object):
    def __init__(self, maze, rat=(1, 1)):
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        self.target = (nrows - 2, ncols - 2)  # target cell where the "cheese" is
        self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == 1.0]
        self.free_cells.remove(self.target)
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not rat in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        self.reset(rat)
        self.num_actions = 4

    def reset(self, rat):
        """Resets instance variables for every episode
        :param rat: current cell
        :return: updates instance variables
        """
        self.rat = rat
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = rat
        self.maze[row, col] = 0.5
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        """Updates next cell location and condition of next cell (blocked, valid, invalid)
        :param action: action taken
        :return: updates self.state
        """
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state

        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))  # mark visited cell

        valid_actions = self.valid_actions()
        LEFT = 0
        UP = 1
        RIGHT = 2
        DOWN = 3
        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:  # invalid action, no change in rat position
            nmode = 'invalid'

        # new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        """Gets reward of current move
        :return: reward value
        """
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows - 2 and rat_col == ncols - 2:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (rat_row, rat_col) in self.visited:
            return -0.25
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04

    def act(self, action):
        """Update state, get reward and check environment state
        :param action: action taken
        :return: next environment state, reward, and status of the game
        """
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        """Get current environment state
        :return: environment state
        """
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0
        # draw the rat
        row, col, valid = self.state
        canvas[row, col] = 0.5
        return canvas

    def game_status(self):
        """check if the game is a win, lose or not over
        :return: string of game status
        """
        if self.total_reward < self.min_reward:
            return 'lose'
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows - 2 and rat_col == ncols - 2:
            return 'win'

        return 'not_over'

    def valid_actions(self, cell=None):
        """Get all valid actions
        :param cell: current cell
        :return: a list of valid actions
        """
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows - 1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols - 1:
            actions.remove(2)

        if row > 0 and self.maze[row - 1, col] == 0.0:
            actions.remove(1)
        if row < nrows - 1 and self.maze[row + 1, col] == 0.0:
            actions.remove(3)

        if col > 0 and self.maze[row, col - 1] == 0.0:
            actions.remove(0)
        if col < ncols - 1 and self.maze[row, col + 1] == 0.0:
            actions.remove(2)

        return actions

class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        """Keep previous state, reward, game status etc. in memory
        :param episode: a list of info at current step
        :return: new info stored in memory
        """
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate == flattened 1d maze cells info, including rat cell (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        """Predicts q values based on current environment state
        :param envstate: current environment state
        :return: a list of four q values
        """
        return self.model.predict(envstate)[0]

    def get_data(self, data_size=10):
        """Gets input and target for training
        :param data_size: batch size
        :return: inputs and targets
        """
        env_size = self.memory[0][0].shape[1]   # envstate 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets


class deepq():
    def __init__(self, maze):
        self.maze = maze
        self.epsilon = 0.1
        self.model = None


    def build_model(self):
        """Build Model Structure
        :return: model
        """
        model = Sequential()
        model.add(Dense(self.maze.size, input_shape=(self.maze.size,)))
        model.add(PReLU())
        model.add(Dense(self.maze.size))
        model.add(PReLU())
        model.add(Dense(4))
        model.compile(optimizer='adam', loss='mse')
        self.model = model

    def qtrain(self, **opt):
        """Start deep q learning with n different starting point.
        If the win rate is 100% in the past n tries, stop the training.
        :param opt: parameters
        :return: model object
        """
        n_epoch = opt.get('n_epoch', 15000)
        max_memory = opt.get('max_memory', 1000)
        data_size = opt.get('data_size', 50)
        weights_file = opt.get('weights_file', "")
        name = opt.get('name', 'model')
        start_time = datetime.datetime.now()

        # If you want to continue training from a previous model,
        # just supply the h5 file name to weights_file option
        if weights_file:
            print("loading weights from file: %s" % (weights_file,))
            self.model.load_weights(weights_file)

        # Construct environment/game from numpy array: maze (see above)
        qmaze = Qmaze(maze)

        # Initialize experience replay object
        experience = Experience(self.model, max_memory=max_memory)

        win_history = []  # history of win/lose game
        # n_free_cells = len(qmaze.free_cells)
        hsize = qmaze.maze.size // 2  # history window size
        win_rate = 0.0
        # imctr = 1

        for epoch in range(n_epoch):
            loss = 0.0
            rat_cell = random.choice(qmaze.free_cells)
            qmaze.reset(rat_cell)
            game_over = False

            # get initial envstate (1d flattened canvas)
            envstate = qmaze.observe()

            n_episodes = 0
            while not game_over:
                valid_actions = qmaze.valid_actions()
                if not valid_actions: break
                prev_envstate = envstate
                # Get next action
                if np.random.rand() < self.epsilon:
                    action = random.choice(valid_actions)
                else:
                    action = np.argmax(experience.predict(prev_envstate))

                # Apply action, get reward and new envstate
                envstate, reward, game_status = qmaze.act(action)
                if game_status == 'win':
                    win_history.append(1)
                    game_over = True
                elif game_status == 'lose':
                    win_history.append(0)
                    game_over = True
                else:
                    game_over = False

                # Store episode (experience)
                episode = [prev_envstate, action, reward, envstate, game_over]
                experience.remember(episode)
                n_episodes += 1

                # Train neural network model
                inputs, targets = experience.get_data(data_size=data_size)
                self.model.fit(
                    inputs,
                    targets,
                    epochs=1,  # 8
                    batch_size=16,
                    verbose=0,
                )
                loss = self.model.evaluate(inputs, targets, verbose=0)

            if len(win_history) > hsize:
                win_rate = sum(win_history[-hsize:]) / hsize

            dt = (datetime.datetime.now() - start_time).total_seconds()/60
            template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
            print(template.format(epoch, n_epoch - 1, loss, n_episodes, sum(win_history), win_rate, dt))
            # we simply check if training has exhausted all free cells and if in all
            # cases the agent won
            if win_rate > 0.9: self.epsilon = 0.05
            if sum(win_history[-hsize:]) == hsize and self.completion_check(qmaze):
                print("Reached 100%% win rate at epoch: %d" % (epoch,))
                break

    def play_game(self, qmaze, rat_cell):
        qmaze.reset(rat_cell)
        envstate = qmaze.observe()
        while True:
            prev_envstate = envstate
            # get next action
            q = self.model.predict(prev_envstate)
            action = np.argmax(q[0])

            # apply action, get rewards and new state
            envstate, reward, game_status = qmaze.act(action)
            if game_status == 'win':
                return True
            elif game_status == 'lose':
                return False

    def completion_check(self, qmaze):
        for cell in qmaze.free_cells:
            if not qmaze.valid_actions(cell):
                return False
            if not self.play_game(qmaze, cell):
                return False
        return True

    def main(self):
        self.build_model()
        self.qtrain(n_epoch=1000, max_memory=8*self.maze.size, data_size=32)


maze =  np.array([
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  0.],
    [0.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,  0.],
    [0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.],
    [0.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,  0.],
    [0.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  0.],
    [0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  0.],
    [0.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]
])

deep_q = deepq(maze)
deep_q.main()