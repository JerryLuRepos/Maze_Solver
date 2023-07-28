from Create_Maze import create_maze
import numpy as np

class flood_maze(create_maze):
    def __init__(self, **create_maze_kwargs):
        super().__init__(**create_maze_kwargs)
        self.wall = 99
        self.nrow = len(self.maze)
        self.ncol = len(self.maze[0])
        self.maze_wall = []
        self.flood_maze = []
        self.wall_arry = []
        self.temp_wall = np.array([
            [self.wall] * self.ncol
            if i == 0 or i == self.nrow - 1
            else [self.wall] + [0] * (self.ncol - 2) + [self.wall]
            for i in range(self.nrow)
        ])

    def fill_wall(self):
        """Fill a value for wall and 0 for others
        :return: update maze_wall
        """
        maze = np.array(self.maze)
        maze_wall = np.array([[0] * self.ncol for _ in range(self.nrow)])
        for row in range(self.nrow):
            for col in range(self.ncol):
                if maze[row, col] == 'w':
                    maze_wall[row, col] = self.wall

        self.maze_wall = maze_wall

    def create_wall_arry(self):
        """for every cell, give a value based on the presence of neighbor walls
        :return: update wall_arry
        """
        maze = np.array(self.maze_wall)
        wall_arry = np.array([[0]*self.ncol for _ in range(self.nrow)])
        for row in range(self.nrow):
            for col in range(self.ncol):
                if maze[row, col] == self.wall:
                    wall_arry[row, col] = -16
                    continue

                up, down, left, right = False, False, False, False
                if row - 1 >= 0 and maze[row - 1, col] == self.wall: up = True
                if row + 1 < self.nrow and maze[row + 1, col] == self.wall: down = True
                if col - 1 >= 0 and maze[row, col - 1] == self.wall: left = True
                if col + 1 < self.ncol and maze[row, col + 1] == self.wall: right = True

                if left and sum([up, down, right]) == 0: wall_arry[row, col] = -1
                elif up and sum([left, down, right]) == 0: wall_arry[row, col] = -2
                elif right and sum([up, down, left]) == 0:wall_arry[row, col] = -3
                elif down and sum([up, left, right]) == 0: wall_arry[row, col] = -4

                elif left and down and sum([up, right]) == 0: wall_arry[row, col] = -5
                elif right and down and sum([up, left]) == 0: wall_arry[row, col] = -6
                elif up and right and sum([down, left]) == 0: wall_arry[row, col] = -7
                elif up and left and sum([down, right]) == 0: wall_arry[row, col] = -8
                elif left and right and sum([up, down]) == 0: wall_arry[row, col] = -9
                elif up and down and sum([left, right]) == 0: wall_arry[row, col] = -10

                elif not up and sum([right, down, left]) == 3: wall_arry[row, col] = -11
                elif not right and sum([up, down, left]) == 3: wall_arry[row, col] = -12
                elif not down and sum([right, up, left]) == 3: wall_arry[row, col] = -13
                elif not left and sum([right, down, up]) == 3: wall_arry[row, col] = -14

                elif sum([up, down, right, left]) == 0: wall_arry[row, col] = -15

        self.wall_arry = wall_arry

    def fill_flood_array(self):
        """do BFS starting from goal location based on agent visited neigh walls
        :return: update flood_maze which is contains steps of every cell to goal
        """
        flood_maze = self.temp_wall.copy()
        visited = []
        que = [[tuple(self.goal_loc)]]
        while que:
            path = que.pop(0)
            cur = path[-1]
            cur_row, cur_col = cur[0], cur[1]
            flood_maze[cur_row, cur_col] = len(path) - 1
            neighs = [(cur_row - 1, cur_col), (cur_row + 1, cur_col), (cur_row, cur_col - 1), (cur_row, cur_col + 1)]
            for next in neighs:
                next_row, next_col = next[0], next[1]
                if flood_maze[next_row, next_col] <= flood_maze[cur_row, cur_col] and next not in visited:
                    que.append(path + [next])
            if cur not in visited:
                visited.append(cur)

        self.flood_maze = flood_maze

    def get_valid_actions(self, cur):
        """Gets all valid actions based on wall_arry
        :param cur: current cell
        :return: a list of all possible next step cell
        """
        wall_idx = self.wall_arry[cur]
        cur_row, cur_col = cur[0], cur[1]
        up, down, left, right = (cur_row - 1, cur_col), (cur_row + 1, cur_col), (cur_row, cur_col - 1), (cur_row, cur_col + 1)
        actions_dict = {
            -1: [up, down, right],
            -2: [left, down, right],
            -3: [up, down, left],
            -4: [up, left, right],
            -5: [up, right],
            -6: [up, left],
            -7: [down, left],
            -8: [down, right],
            -9: [up, down],
            -10: [left, right],
            -11: [up],
            -12: [right],
            -13: [down],
            -14: [left],
            -15: [up, down, left, right],
            -16: []
        }
        return actions_dict[wall_idx]

    def get_current_wall(self, cur):
        """Updates wall info based on current cell
        :param cur: current cell
        :return: updates temp_wall
        """
        cur_row, cur_col = cur[0], cur[1]
        up, down, left, right = (cur_row - 1, cur_col), (cur_row + 1, cur_col), (cur_row, cur_col - 1), (cur_row, cur_col + 1)
        self.temp_wall[up] = self.maze_wall[up]
        self.temp_wall[down] = self.maze_wall[down]
        self.temp_wall[left] = self.maze_wall[left]
        self.temp_wall[right] = self.maze_wall[right]

    def find_goal(self):
        """Run 5 loops of the following:
        1. get all valid actions
        2. update current wall info
        3. do BFS and find all step counts for each cell based on updated wall info
        4. get next step based on minimum step counts in neighbor cells in flood_maze
        5. repeat 1-4 until goal location is found.
        :return:
        """
        all_steps = []
        for i in range(5):
            cur = tuple(self.start_point)
            all_steps = [cur]
            while cur != tuple(self.goal_loc):
                valid_actions = self.get_valid_actions(cur)
                self.get_current_wall(cur)
                self.fill_flood_array()
                flood_min_idx = np.argmin([self.flood_maze[step] for step in valid_actions])
                next_step = valid_actions[flood_min_idx]
                all_steps.append(next_step)
                cur = next_step

        maze = np.array(self.maze)
        for step in all_steps:
            maze[step] = 'A'
        for row in maze:
            print(row)

        return all_steps

    def main(self):
        self.fill_wall()
        self.create_wall_arry()
        self.find_goal()


if __name__ == '__main__':
    get_maze = create_maze(16, 16, seed=1)
    get_maze.add_goal()
    get_maze.create_valid_path(visualize=False)
    get_maze.create_walls(max_cycle_len=0, visualize=True)
    create_maze_kwargs = vars(get_maze)

    flood = flood_maze(**create_maze_kwargs)
    flood.main()