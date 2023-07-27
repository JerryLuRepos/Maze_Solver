---
# Maze_Solver

## Purpose
The purpose of this repo is to experiment different methods for solving a maze. It is inspired by the Micromouse contest which is an event where small robotic mice compete to solve a 16×16 maze as quick as possible with a goal at the center of the maze. 

## Methods
1. Value Based Q Learning
2. Deep Q Learning
3. Flood Fill Algorithm

All three methods can be used to solve a maze from a starting point to the goal location. However the amount of time taken for each method to solve a maze may vary.

## Conclusion
After experimenting on all three methods... 
The FloodFill algorithm takes the least time to solve a maze and is able to solve the maze without complete information about the maze.
The standard q learning is really fast as well but may require additional model training to be able to solve the maze without complete information.
The Deep Q Learning takes a lot of time to train due to its self retraining process. Clearly it's not a good fit for the Micromouse contest.

