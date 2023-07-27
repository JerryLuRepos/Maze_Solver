from random import *
import numpy as np

class create_maze:
    def __init__(self, row=16, col=16, maze=[], corners=[],
                 goal_loc=[], start_point=[], valid_path=[], seed=1):
        self.row = row
        self.col = col
        # create empty maze
        if not maze:
            self.maze = [
                ['w'] * self.col
                if i == 0 or i == self.row - 1
                else ['w'] + ['.'] * (self.col - 2) + ['w']
                for i in range(self.row)
            ]
        else: self.maze = maze
        # define corners
        if not corners:
            self.corners = [
                [1,1],
                [1,len(self.maze[0])-2],
                [len(self.maze)-2,1],
                [len(self.maze)-2,len(self.maze[0])-2],
            ]
        else: self.corners = corners
        self.goal_loc = goal_loc
        self.start_point = start_point
        self.valid_path = valid_path
        self.seed = Random(seed)

    ####################
    # Helper Functions #
    ####################
    def _creates_neighbor_lst(self, matrix, cur, result, num=4):
        """creates a list of neighbor coordinates or values
        :param matrix: input data to search neighbors at
        :param cur: current coordinates
        :param result: type of output: coordinates or values
        :param num: number of neighbors. 4: up, left, right, down 9:ul, u, ur, l, current, r, ll, l, lr
        :return: a list of results
        """
        row, col = cur[0], cur[1]
        neighs = []
        for rdelta in [-1, 0, 1]:
            for cdelta in [-1, 0, 1]:
                n_row = max(0, min(self.row - 1, row + rdelta))
                n_col = max(0, min(self.col - 1, col + cdelta))
                neighs.append([n_row, n_col])

        if num == 4:
            neighs = [neighs[i] for i in [1, 3, 5, 7]]

        output = [] # find unique coords
        for i in neighs:
            if i not in output:
                output.append(i)

        if result == 'coordinates':
            return output
        if result == 'values':
            return [matrix[row][col] for row, col in output]

    def _find_overlap_grid9(self, cur, neigh, grid9):
        """Finds overlaps of 9grid neighbor between current and neighbor coords
        :param cur: current coord
        :param neigh: next neigh coord
        :param grid9: grid 9 neighbors of current coord
        :return: a list of overlapped coords between current and next neighbor coord
        """
        row_delta = cur[0] - neigh[0]
        col_delta = cur[1] - neigh[1]
        grid9 = np.reshape(grid9, (3,3,2))
        overlap = [list(grid9[row][col])
                   for row in range(max([0,0+row_delta]), min([3,3+row_delta]))
                   for col in range(max([0,0+col_delta]), min([3,3+col_delta]))]

        return overlap

    def _find_edge_cycle(self, path_to_edge):
        """Finds length of cycle with edge. If multiple cycles exist, get the maximum length
        :param path_to_edge: all path from one coord that connects to edges
        :return: maximum cycle length with edges (if there are multiple possible cycles)
        """
        def find_shortest_edge_path(cur, ct=0, hist={}):
            """Finds minimum length of edge between two coords
            """
            row, col = cur[0], cur[1]
            if cur == end:
                return ct

            if (row, col) in hist:
                return hist[(row, col)]
            else:
                hist[(row, col)] = ct

            min_length = float('inf')
            for neigh in self._creates_neighbor_lst(self.maze, cur, 'coordinates', 4):
                n_row, n_col = neigh[0], neigh[1]
                if n_row == 0 or n_row == len(self.maze) - 1 or n_col == 0 or n_col == len(self.maze[0]) - 1 or neigh == end:
                    min_length = min(min_length, find_shortest_edge_path(neigh, ct+1, hist))

            return min_length

        cycle_lengths = [] #contains all cycle length
        # loop through all combinations of coords that are next to edges
        for idx1 in range(0, len(path_to_edge)-1):
            for idx2 in range(idx1+1, len(path_to_edge)):
                start, end = path_to_edge[idx1][-1], path_to_edge[idx2][-1]
                edge_length = find_shortest_edge_path(start)
                cycle_lengths.append(edge_length + len(path_to_edge[idx1]) + len(path_to_edge[idx2]))

        return max(cycle_lengths)

    def _len_cycle_path_to_edge(self, start):
        """Finds length of inner cycle and all path to edge.
        :param start: starting coord
        :return: maximum cycle length and length to edge
        """
        inner_cycles = []
        path_to_edge = []
        def inner_cycle_and_path_to_edge(cur, path=[], overlap_coords=[], hist={}):
            """append all inner cycle and path to edge.
            """
            row, col = cur[0], cur[1]
            # append any inner cycles
            if (row, col) in hist and cur == start:
                if not inner_cycles or len(path) < len(inner_cycles[-1]):
                    inner_cycles.append(path)
                return
            # append any path to edge
            if row == 0 or row == len(self.maze) - 1 or col == 0 or col == len(self.maze[0]) - 1:
                if path not in path_to_edge:
                    path_to_edge.append(path)
                return

            if (row, col) in hist:
                return
            else:
                hist[(row, col)] = 1

            neigh9 = self._creates_neighbor_lst(self.maze, cur, 'coordinates', 9)
            for neigh in neigh9:
                n_row, n_col = neigh[0], neigh[1]
                if neigh not in overlap_coords and neigh != cur and (self.maze[n_row][n_col] == 'w' or neigh == start):
                    inner_cycle_and_path_to_edge(neigh, path + [cur], self._find_overlap_grid9(cur, neigh, neigh9), hist)

            return

        inner_cycle_and_path_to_edge(start)

        max_cycle_length = 0
        for cycle in inner_cycles:
            max_cycle_length = max(max_cycle_length, len(cycle)) # find maximum inner cycle length
        if len(path_to_edge) >= 2:
            max_cycle_length = max(max_cycle_length, self._find_edge_cycle(path_to_edge)) # update max length if edge cycles exist
        # find max length of path to edge
        max_path_edge_len = 0
        if len(path_to_edge) > 0:
            max_path_edge_len = max([len(i) for i in path_to_edge])

        return max_cycle_length, max_path_edge_len


    #############################################

    # Step 1
    def add_goal(self):
        """Adds goal to the middle of maze with three walls
        :return:
            update self.maze with G and ww
            update self.goal_loc
        """
        goal_row, goal_col = self.row//2, self.col//2
        self._creates_neighbor_lst(self.maze, [goal_row, goal_col], 4)
        goal_walls = self._creates_neighbor_lst(self.maze, [goal_row, goal_col], 'coordinates', 4)
        goal_walls.pop(self.seed.randint(0, 3)) # randomly remove one wall
        self.maze[goal_row][goal_col] = 'G'
        self.goal_loc = [goal_row, goal_col]
        for wall in goal_walls:
            self.maze[wall[0]][wall[1]] = 'w'

    # Step 2
    def create_valid_path(self, visualize=True):
        """Finds a valid path from one of the maze corner to the goal_loc to ensure it's a valid maze
        :param visualize: True or False. Prints out the valid path in maze
        :return:
            self.valid_path. A list of coordinates from goal_loc to starting point
            self.start_point. coordinates of starting point
        """
        def find_path(path, hist_path={}):
            """finds a valid path"""
            cur = path[-1]
            row, col = cur[0], cur[1]
            # check if the current coordinates are at maze corners
            if cur in self.corners:
                self.start_point = cur
                return path

            if (row, col) in hist_path: # if current is in memory return
                return None
            neigh4_coords = self._creates_neighbor_lst(self.maze, [row, col], 'coordinates', 4)
            if len([[r, c] for r, c in neigh4_coords if (r, c) in hist_path]) >= 2:
                # check the number of neighbor coordinates in history path
                # if >= 2, the path creates an area instead of line of path
                return None
            # memorization
            hist_path[(row, col)] = 1

            self.seed.shuffle(neigh4_coords) # shuffle the directions to randomize path
            final_path = None
            for direction_coord in neigh4_coords:
                n_row, n_col = direction_coord[0], direction_coord[1]
                if 1 <= n_row <= len(self.maze)-2 and 1 <= n_col <= len(self.maze[0])-2 and self.maze[n_row][n_col] != 'w':
                    final_path = find_path(path+[direction_coord], hist_path)

                if final_path is not None: # return the first path that goes to the corner
                    return final_path

            return final_path

        self.valid_path = find_path([self.goal_loc])

        # visualize valid path in maze
        if visualize:
            print("Visualize Random Valid Path")
            valid_path = sorted(self.valid_path[1:], key=lambda x: x[0])
            row_ct = 0
            while row_ct < len(self.maze):
                cur_row = self.maze[row_ct].copy()
                while valid_path and valid_path[0][0] == row_ct:
                    cur_row[valid_path[0][1]] = 'p'
                    valid_path.pop(0)
                row_ct += 1
                print(cur_row)

    # Step 3
    def create_walls(self, max_cycle_len=4, visualize=True):
        """Creates walls for maze
        :param max_cycle_len: longest length of path that creates an area. increase to create more and longer walls
        :param visualize: True or False
        :return: update self.maze for walls
        """
        def create_walls_bfs(cur_row, cur_col, visited={}):
            """Using bfs to check and create valid walls."""
            if (cur_row, cur_col) in visited or self.maze[cur_row][cur_col] in ['w','G']:
                return

            if (cur_row, cur_col) in visited:
                return
            else:
                visited[(cur_row, cur_col)] = 1

            if [cur_row, cur_col] not in self.valid_path:
                cycle_len, path_edge_len = self._len_cycle_path_to_edge([cur_row, cur_col])
                if cycle_len <= max_cycle_len and self.seed.randint(1,100) <= 100/(path_edge_len+1):
                    self.maze[cur_row][cur_col] = 'w'
                    return

            neighs = self._creates_neighbor_lst(self.maze, [cur_row, cur_col], 'coordinates', 4)
            self.seed.shuffle(neighs)
            for row, col in neighs:
                create_walls_bfs(row, col, visited)

        create_walls_bfs(self.start_point[0], self.start_point[1])

        if visualize:
            print('Visualize {} by {} Maze Starting at {}'.format(self.row, self.col, self.start_point))
            for row in self.maze:
                print(row)



# for i in range(5):
#     get_maze = create_maze(16, 16, seed=1)
#     get_maze.add_goal()
#     get_maze.create_valid_path(visualize=False)
#     get_maze.create_walls(max_cycle_len=0, visualize=True)