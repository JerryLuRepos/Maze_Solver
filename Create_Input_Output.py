from Q_Learning import q_learning
from Create_Maze import create_maze

class creat_input_output():
    def __init__(self, maze_x=[], maze_y=[]):
        self.maze_x = maze_x
        self.maze_y = maze_y


    def agent_vision(self, maze, q_values, q_shortest_path):
        def walls_in_vision(cur, prev_maze):
            row, col = cur[0], cur[1]
            all_walls = []
            # left vision
            move_left = -1
            while 0 <= col + move_left < len(maze[0]) and maze[row][col + move_left] != 'w':
                move_left -= 1
            all_walls.append([row, col + move_left])
            # right vision
            move_right = 1
            while 0 <= col + move_right < len(maze[0]) and maze[row][col + move_right] != 'w':
                move_right += 1
            all_walls.append([row, col + move_right])
            # up vision
            move_up = -1
            while 0 <= row + move_up < len(maze) and maze[row + move_up][col] != 'w':
                move_up -= 1
            all_walls.append([row + move_up, col])
            # down vision
            move_down = 1
            while 0 <= row + move_down < len(maze) and maze[row + move_down][col] != 'w':
                move_down += 1
            all_walls.append([row + move_down, col])
            # up left vision
            move_up, move_left = -1, -1
            while 0 <= col + move_left < len(maze[0]) and 0 <= row + move_up < len(maze) and maze[row + move_up][col + move_left] != 'w':
                move_up -= 1
                move_left -= 1
            all_walls.append([row + move_up, col + move_left])
            # up right vision
            move_up, move_right = -1, 1
            while 0 <= col + move_right < len(maze[0]) and 0 <= row + move_up < len(maze) and maze[row + move_up][col + move_right] != 'w':
                move_up -= 1
                move_right += 1
            all_walls.append([row + move_up, col + move_right])
            # down left vision
            move_down, move_left = 1, -1
            while 0 <= col + move_left < len(maze[0]) and 0 <= row + move_down < len(maze) and maze[row + move_down][col + move_left] != 'w':
                move_down += 1
                move_left -= 1
            all_walls.append([row + move_down, col + move_left])
            # down right vision
            move_down, move_right = 1, 1
            while 0 <= col + move_right < len(maze[0]) and 0 <= row + move_down < len(maze) and maze[row + move_down][col + move_right] != 'w':
                move_down += 1
                move_right += 1
            all_walls.append([row + move_down, col + move_right])

            for wall in all_walls:
                prev_maze[wall[0]][wall[1]] = 'w'

            return prev_maze


        prev_maze = [
            ['w'] * len(maze[0])
            if i == 0 or i == len(maze) - 1
            else ['w'] + ['.'] * (len(maze[0]) - 2) + ['w']
            for i in range(len(maze))
        ]
        maze_x = []
        maze_y = []
        for idx in range(len(q_shortest_path[:-1])):
            step = q_shortest_path[idx]
            cur_maze = walls_in_vision(step, prev_maze)
            # list is an object so it is stored by reference.
            # continue updating list A (e.g. A[0] = 1) and append into list B will make list B contains the the same copy of the last update of list A
            # So make a copy of the list that had updates. do cur_maze.copy doesn't work because it's the list inside that had updates
            cur_maze = [row.copy() for row in cur_maze]
            cur_maze[step[0]][step[1]] = 'A'
            maze_x.append(cur_maze)
            maze_y.append(q_values[step[0], step[1]])

        return maze_x, maze_y


# get_maze = create_maze(16, 16, seed=1)
# get_maze.add_goal()
# get_maze.create_valid_path(visualize=False)
# get_maze.create_walls(max_cycle_len=0, visualize=False)
# create_maze_kwargs = vars(get_maze)
#
# qlearn = q_learning(**create_maze_kwargs)
# qlearn.convert_maze_to_rewards()
# qlearn.q_train()
# start_point = qlearn.start_point
# start_row, start_col = start_point[0], start_point[1]
# qlearn.get_shortest_path(start_row, start_col)
# qlearning_kwargs = vars(qlearn)
#
# get_data = creat_input_output()
# get_data.agent_vision(qlearning_kwargs['maze'], qlearning_kwargs['q_values'], qlearning_kwargs['q_shortest_path'])
