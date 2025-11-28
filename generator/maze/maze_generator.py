import random
from typing import List, Tuple, Optional

def generate_maze(width: int, height: int, seed: Optional[int] = None) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """
    Generate a random maze using DFS. 0 = open, 1 = wall.
    Returns maze grid and solution path from (0,0) to (width-1,height-1).
    """
    # Ensure odd dimensions for proper maze
    # allow reproducible randomness via seed
    rnd = random.Random(seed)

    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1
    maze = [[1 for _ in range(width)] for _ in range(height)]
    def in_maze(nx, ny):
        return 0 <= nx < width and 0 <= ny < height
    def neighbors(x, y):
        # return neighbors in randomized order to increase diversity
        dirs = [(-2,0),(2,0),(0,-2),(0,2)]
        rnd.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x+dx, y+dy
            if in_maze(nx, ny):
                yield nx, ny
    stack = [(1, 1)]
    maze[1][1] = 0
    visited = set([(1, 1)])
    while stack:
        x, y = stack[-1]
        nbs = [(nx, ny) for nx, ny in neighbors(x, y) if (nx, ny) not in visited]
        if nbs:
            nx, ny = rnd.choice(nbs)
            maze[(y+ny)//2][(x+nx)//2] = 0  # Carve wall between
            maze[ny][nx] = 0
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()
    # Find solution path using BFS
    from collections import deque
    queue = deque([((1,1), [(1,1)])])
    visited_bfs = set([(1,1)])
    solution = []
    while queue:
        (x, y), cur_path = queue.popleft()
        if (x, y) == (width-2, height-2):
            solution = cur_path
            break
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if in_maze(nx, ny) and maze[ny][nx] == 0 and (nx, ny) not in visited_bfs:
                visited_bfs.add((nx, ny))
                queue.append(((nx, ny), cur_path + [(nx, ny)]))
    # Try to connect dead zones to main path, but keep only one solution
    reachable = visited_bfs.copy()
    # Randomize dead-zone handling order to increase variability
    coords = [(x, y) for y in range(height) for x in range(width)]
    rnd.shuffle(coords)
    for x, y in coords:
        if maze[y][x] == 0 and (x, y) not in reachable:
            # Try to connect to reachable area with a randomized neighbor order
            nbrs = [(x+dx, y+dy) for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]]
            rnd.shuffle(nbrs)
            connected = False
            for nx, ny in nbrs:
                if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 0 and (nx, ny) in reachable:
                    maze[y][x] = 0
                    if _has_multiple_solutions(maze, (1,1), (width-2,height-2)):
                        maze[y][x] = 1
                    else:
                        reachable.add((x, y))
                        connected = True
                    break
            if not connected:
                maze[y][x] = 1
    # Add extra branches (dead ends) to increase possible paths
    # randomize number of extra branch attempts to vary density
    extra_branch_attempts = rnd.randint(max(1, int(width * height * 0.05)), max(1, int(width * height * 0.25)))
    for _ in range(extra_branch_attempts):
        # Randomly pick a wall adjacent to a path
        candidates = []
        for y in range(1, height-1):
            for x in range(1, width-1):
                if maze[y][x] == 1:
                    adj_paths = [(x+dx, y+dy) for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)] if maze[y+dy][x+dx] == 0]
                    # accept candidates with 1 or 2 adjacent paths to increase branch patterns
                    if 1 <= len(adj_paths) <= 2:
                        candidates.append((x, y, adj_paths[0]))
        if candidates:
            x, y, (px, py) = rnd.choice(candidates)
            maze[y][x] = 0
            # Randomly keep or revert based on multiple-solution check and a small probability to allow rare multiple solutions
            if _has_multiple_solutions(maze, (1,1), (width-2,height-2)):
                # small chance to keep multiple solutions for diversity
                if rnd.random() < 0.05:
                    pass
                else:
                    maze[y][x] = 1
    return maze, solution

def _has_multiple_solutions(maze, start, end):
    """Return True if maze has more than one solution from start to end."""
    from collections import deque
    height = len(maze)
    width = len(maze[0])
    queue = deque([(start, [start])])
    found = 0
    visited_paths = set()
    while queue:
        (x, y), cur_path = queue.popleft()
        if (x, y) == end:
            found += 1
            if found > 1:
                return True
            continue
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 0 and (nx, ny) not in cur_path:
                queue.append(((nx, ny), cur_path + [(nx, ny)]))
    return False
def print_maze(maze: List[List[int]], solution: List[Tuple[int, int]] = []):
    height = len(maze)
    width = len(maze[0])
    sol_set = set(solution) if solution else set()
    for y in range(height):
        row = ''
        for x in range(width):
            if (x, y) == (1, 1):
                row += 'S'
            elif (x, y) == (width-2, height-2):
                row += 'E'
            elif (x, y) in sol_set:
                row += '.'
            elif maze[y][x] == 1:
                row += '#'
            else:
                row += ' '
        print(row)

if __name__ == "__main__":
    maze, solution = generate_maze(25, 25)
    print("Generated maze:")
    print_maze(maze, solution)
    print("\nSolution path:")
    print(solution)
