from Create_Maze import create_maze
from Q_Learning import q_learning
from Create_Input_Output import creat_input_output
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizer_v2.adam import Adam
from keras.layers.advanced_activations import PReLU, ReLU
import numpy as np

def build_model():
    # construct model
    model = Sequential()
    model.add(Dense(256, input_shape=(256,)))
    model.add(ReLU())
    model.add(Dense(256))
    model.add(ReLU())
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

def maze_conversion(maze):
    """Converts maze into rewards table
    :return: numpy array of rewards
    """
    default_state = []
    for i in range(len(maze)):
        row = []
        for j in range(len(maze[0])):
            cur = maze[i][j]
            if cur == 'w':
                row.append(0)
            elif cur == 'A':
                row.append(0.5)
            else:
                row.append(1)
        default_state.append(np.array(row))
    return np.array(default_state)

model = build_model()

for seed in range(100):
    get_maze = create_maze(16, 16, seed=seed)
    get_maze.add_goal()
    get_maze.create_valid_path(visualize=False)
    get_maze.create_walls(max_cycle_len=0, visualize=False)
    create_maze_kwargs = vars(get_maze)

    qlearn = q_learning(**create_maze_kwargs)
    qlearn.convert_maze_to_rewards()
    qlearn.q_train()
    start_point = qlearn.start_point
    start_row, start_col = start_point[0], start_point[1]
    qlearn.get_shortest_path(start_row, start_col)
    qlearning_kwargs = vars(qlearn)

    get_data = creat_input_output()
    maze_x, maze_y = get_data.agent_vision(qlearning_kwargs['maze'], qlearning_kwargs['q_values'], qlearning_kwargs['q_shortest_path'])
    for idx in range(len(maze_x)):
        inputs = maze_conversion(maze_x[idx]).reshape(1, -1)
        targets = np.array([[1 if i == np.argmax(maze_y[idx]) else 0 for i in range(4)]])
        model.fit(
            inputs,
            targets,
            epochs=5,
            verbose=0,
        )
        loss = model.evaluate(inputs, targets, verbose=0)
        print('Maze {} Step {} Path Len {} Loss: {}'.format(seed, idx, len(maze_x), loss))


get_maze = create_maze(16, 16, seed=1001)
get_maze.add_goal()
get_maze.create_valid_path(visualize=False)
get_maze.create_walls(max_cycle_len=0, visualize=False)