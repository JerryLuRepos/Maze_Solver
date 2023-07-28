import numpy as np
from Create_Maze import create_maze
import json


class q_learning(create_maze):
    def __init__(self, **create_maze_kwargs):
        super().__init__(**create_maze_kwargs)
        self.rewards = None
        self.q_values = np.zeros((self.row, self.col, 4))
        self.q_shortest_path = []
        self.bfs_shortest_path = []
        self.configs = json.load(open('config.json', 'r'))['q']

    def convert_maze_to_rewards(self):
        """Converts maze into rewards table
        :return: numpy array of rewards
        """
        wall_penalty = self.configs['wall_penalty']
        goal_reward = self.configs['goal_reward']
        step_penalty = self.configs['step_penalty']

        reward = []
        for i in range(len(self.maze)):
            row = []
            for j in range(len(self.maze[0])):
                cur = self.maze[i][j]
                if cur == 'w':
                    row.append(wall_penalty)
                elif cur == 'G':
                    row.append(goal_reward)
                else:
                    row.append(step_penalty)
            reward.append(np.array(row))

        self.rewards = np.array(reward)

    def check_invalid_state(self, row, col):
        """Checks if current location is invalid
        :param row: row idx
        :param col: col idx
        :return: True or False
        """
        if self.rewards[row, col] == -0.5:
            return False
        else:
            return True

    def get_epsilon(self, episode):
        """Calculates Epsilon Decay
        :param episode: number of path ran
        :return: Decay Epsilon
        """
        epsilon = self.configs['epsilon']
        decay_rate = self.configs['epsilon_decay']
        min_epsilon = self.configs['min_epsilon']
        return max(min_epsilon, epsilon*(decay_rate**episode))

    def get_random_starting_location(self):
        """randomly selects a valid starting location
        :return: starting row and column idx
        """
        row_idx = np.random.randint(self.row)
        col_idx = np.random.randint(self.col)

        while self.check_invalid_state(row_idx, col_idx):
            row_idx = np.random.randint(self.row)
            col_idx = np.random.randint(self.col)

        return row_idx, col_idx

    def get_next_action(self, current_row_idx, current_col_idx, epsilon):
        """Uses epsilon to decide if the next action is randomly choosen or picked based on max of q value
        :param current_row_index: current row
        :param current_column_index: current col
        :param epsilon: epsilon
        :return: idx of next action (0:up 1:right 2:down 3:left)
        """
        if np.random.random() < epsilon:
            return np.random.randint(4)
        else:
            return np.argmax(self.q_values[current_row_idx, current_col_idx])

    #define a function that will get the next location based on the chosen action
    def get_next_location(self, current_row_idx, current_col_idx, action_index):
        """Get location of next action
        :param current_row_index: row
        :param current_column_index: col
        :param action_index: action idx 0,1,2,3
        :return:
        """
        new_row_index = current_row_idx
        new_column_index = current_col_idx
        if action_index == 0 and current_row_idx > 0:
            new_row_index -= 1
        elif action_index == 1 and current_col_idx < self.col - 1:
            new_column_index += 1
        elif action_index == 2 and current_row_idx < self.row - 1:
            new_row_index += 1
        elif action_index == 3 and current_col_idx > 0:
            new_column_index -= 1

        return new_row_index, new_column_index

    def get_shortest_path(self, start_row_index, start_column_index):
        """Get shortest path based on max q values
        :param start_row_index: row
        :param start_column_index: col
        :return: a list of location on shortest path
        """
        if self.check_invalid_state(start_row_index, start_column_index):
            return []

        cur_row, cur_col = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([cur_row, cur_col])
        while not self.check_invalid_state(cur_row, cur_col):
            action_index = self.get_next_action(cur_row, cur_col, 0)
            cur_row, cur_col = self.get_next_location(cur_row, cur_col, action_index)
            if [cur_row, cur_col] not in shortest_path:
                shortest_path.append([cur_row, cur_col])
            else:
                print('Loop Detected!')
                break
        self.q_shortest_path = shortest_path

    def find_shortest_path_bfs(self, visualize=False):
        """Finds shortest path using BFS. For double checking if the q values produces the true shortest path
        :param visualize: Visualize shortest path in maze
        :return: shortest path in list
        """
        def search_shortest_path(cur):
            visited = []
            que = [[cur]]
            while que:
                path = que.pop(0)
                cur = path[-1]
                if cur not in visited:
                    for neigh in self._creates_neighbor_lst(self.maze, cur, 'coordinates', 4):
                        n_row, n_col = neigh[0], neigh[1]
                        if self.maze[n_row][n_col] == '.':
                            new_path = path.copy()
                            new_path += [neigh]
                            que.append(new_path)
                        elif self.maze[n_row][n_col] == 'G':
                            return path + [neigh]
                    visited.append(cur)

        shortest_path = search_shortest_path(self.start_point)
        self.bfs_shortest_path = shortest_path

        if visualize:
            print("Visualize Shortest Path")
            valid_path = sorted(shortest_path[1:], key=lambda x: x[0])
            row_ct = 0
            while row_ct < len(self.maze):
                cur_row = self.maze[row_ct].copy()
                while valid_path and valid_path[0][0] == row_ct:
                    cur_row[valid_path[0][1]] = 'p'
                    valid_path.pop(0)
                row_ct += 1

    def q_train(self):
        """Starts q training
        :return: updated q_values
        """
        discounted_rate = self.configs['discounted_rate']
        learning_rate = self.configs['learning_rate']
        n_episodes = self.configs['n_episodes']

        for episode in range(n_episodes):
            row_index, column_index = self.get_random_starting_location()
            epsilon = self.get_epsilon(episode)
            steps = 0
            while not self.check_invalid_state(row_index, column_index):
                action_index = self.get_next_action(row_index, column_index, epsilon)
                old_row_index, old_column_index = row_index, column_index #store the old row and column indexes
                row_index, column_index = self.get_next_location(row_index, column_index, action_index)
                reward = self.rewards[row_index, column_index]
                old_q_value = self.q_values[old_row_index, old_column_index, action_index]
                temporal_difference = reward + (discounted_rate * np.max(self.q_values[row_index, column_index])) - old_q_value
                new_q_value = old_q_value + (learning_rate * temporal_difference) # it's the same as bellman equation: old_state*(1-lr) + lr(reward + dr*max(next_state))
                self.q_values[old_row_index, old_column_index, action_index] = new_q_value
                steps += 1


if __name__ == '__main__':
    get_maze = create_maze(16, 16, seed=2)
    get_maze.add_goal()
    get_maze.create_valid_path(visualize=False)
    get_maze.create_walls(max_cycle_len=0, visualize=False)
    create_maze_kwargs = vars(get_maze)

    qlearn = q_learning(**create_maze_kwargs)
    qlearn.convert_maze_to_rewards()

    # Validation
    success_ct = 0
    for epoch in range(100):
        print('Epoch {}'.format(epoch))
        qlearn.q_train()
        start_point = qlearn.start_point
        start_row, start_col = start_point[0], start_point[1]
        qlearn.get_shortest_path(start_row, start_col)
        qlearn.find_shortest_path_bfs()
        if len(qlearn.q_shortest_path) == len(qlearn.bfs_shortest_path):
            success_ct += 1

    print(success_ct/100)
