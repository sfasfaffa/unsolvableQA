import random
from maze_generator import generate_maze, print_maze
from typing import List, Tuple
import copy

def check_maze_solvable(maze: List[List[int]]) -> bool:
    """Return True if maze is solvable from (0,0) to (width-1,height-1)"""
    from collections import deque
    height = len(maze)
    width = len(maze[0])
    start = (1, 1)
    end = (width-2, height-2)
    queue = deque([start])
    visited = set([start])
    while queue:
        x, y = queue.popleft()
        if (x, y) == end:
            return True
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return False

def add_unsolvable_obstacle(maze: List[List[int]], solution: List[Tuple[int,int]], difficulty: int) -> Tuple[List[List[int]], Tuple[int,int]]:
    """
    Find all possible paths, then block a point (not start/end) on a selected path according to difficulty.
    Returns new maze and obstacle position.
    """
    from collections import deque
    height = len(maze)
    width = len(maze[0])
    # Find all paths using BFS (returns list of paths)
    def all_paths():
        paths = []
        start = (1, 1)
        end = (width-2, height-2)
        queue = deque([(start, [start])])
        while queue:
            (x, y), cur_path = queue.popleft()
            if (x, y) == end:
                paths.append(cur_path)
                continue
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 0 and (nx, ny) not in cur_path:
                    queue.append(((nx, ny), cur_path + [(nx, ny)]))
        return paths
    paths = all_paths()
    if not paths or all(len(p) <= 2 for p in paths):
        return maze, None
    # Select a path according to difficulty
    path_idx = min(len(paths)-1, max(0, int(len(paths)*difficulty/10)))
    chosen_path = paths[path_idx]
    # Select a point on the path (not start/end)
    if len(chosen_path) <= 2:
        return maze, None
    pt_idx = min(len(chosen_path)-2, max(1, int(len(chosen_path)*difficulty/10)))
    obstacle_pos = chosen_path[pt_idx]
    maze_new = copy.deepcopy(maze)
    maze_new[obstacle_pos[1]][obstacle_pos[0]] = 1
    return maze_new, obstacle_pos

def construct_unsolvable_maze(width: int, height: int, difficulty: int = 5, max_attempts: int = 10):
    for attempt in range(max_attempts):
        maze, solution = generate_maze(width, height)
        print(f"maze generated on attempt {attempt+1},maze:{maze}")
        if not solution or len(solution) < 3:
            continue
        maze_unsolvable, obstacle_pos = add_unsolvable_obstacle(maze, solution, difficulty)
        if not obstacle_pos:
            continue
        if not check_maze_solvable(maze_unsolvable):
            print(f"Unsolvable maze generated on attempt {attempt+1}.")
            print_maze(maze_unsolvable)
            print(f"Obstacle added at {obstacle_pos}.")
            return maze_unsolvable, obstacle_pos, maze, solution
        else:
            print(f"Attempt {attempt+1}: Maze still solvable after obstacle. Reflecting...")
            # Reflection: why not unsolvable? Most likely, alternative path exists.
            print("Reflection: The obstacle did not block all possible paths. Try placing obstacle at a more critical point or increase difficulty.")
    print("Failed to generate unsolvable maze after max attempts.")
    return None, None, None, None

if __name__ == "__main__":
    maze_unsolvable, obstacle_pos, maze, solution = construct_unsolvable_maze(8, 8, difficulty=7)
    if maze_unsolvable:
        print("Original maze and solution:")
        print_maze(maze, solution)
        print("\nUnsolvable maze:")
        print_maze(maze_unsolvable)
        print(f"Obstacle at {obstacle_pos}")
    else:
        print("No unsolvable maze generated.")
