import json

def save_solvable_maze_to_text(solvable_maze, solvable_solution, filename="solvable_maze.txt"):
    """
    Save the solvable maze and its solution to a plain text file.

    Args:
        solvable_maze (list): The solvable maze grid.
        solvable_solution (list): The solution path for the solvable maze.
        filename (str): The name of the text file to save the data.
    """
    with open(filename, "w") as f:
        f.write("Solvable Maze:\n")
        for row in solvable_maze:
            f.write("".join(str(cell) for cell in row) + "\n")
        f.write("\nSolvable Solution:\n")
        f.write(" -> ".join(f"({x},{y})" for x, y in solvable_solution) + "\n")

def save_unsolvable_maze_to_text(unsolvable_maze, obstacle_pos, filename="unsolvable_maze.txt"):
    """
    Save the unsolvable maze and its obstacle position to a plain text file.

    Args:
        unsolvable_maze (list): The unsolvable maze grid.
        obstacle_pos (tuple): The position of the obstacle in the unsolvable maze.
        filename (str): The name of the text file to save the data.
    """
    with open(filename, "w") as f:
        f.write("Unsolvable Maze:\n")
        for row in unsolvable_maze:
            f.write("".join(str(cell) for cell in row) + "\n")
        f.write("\nObstacle Position:\n")
        f.write(f"{obstacle_pos}\n")

def save_mazes_to_json_as_text(solvable_maze, solvable_solution, unsolvable_maze, obstacle_pos, filename="mazes.json"):
    """
    Save solvable and unsolvable mazes along with their solutions to a JSON file as text strings.

    Args:
        solvable_maze (list): The solvable maze grid.
        solvable_solution (list): The solution path for the solvable maze.
        unsolvable_maze (list): The unsolvable maze grid.
        obstacle_pos (tuple): The position of the obstacle in the unsolvable maze.
        filename (str): The name of the JSON file to save the data.
    """
    data = {
        "solvable_maze": "\n".join("".join(str(cell) for cell in row) for row in solvable_maze),
        "solvable_solution": " -> ".join(f"({x},{y})" for x, y in solvable_solution),
        "unsolvable_maze": "\n".join("".join(str(cell) for cell in row) for row in unsolvable_maze),
        "obstacle_position": f"{obstacle_pos}"
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def save_maze_to_jsonl_format(data_source, maze_text, solution_text, split, idx, filename="maze_output.jsonl"):
    """
    Save maze data in the required JSONL format.

    Args:
        data_source (str): The source of the data.
        maze_text (str): The maze represented as a string.
        solution_text (str): The solution to the maze, represented as a string.
        split (str): The data split (e.g., train, test).
        idx (int): The index of the data point.
        filename (str): The name of the JSONL file to save the data.
    """
    full_prompt = (
        "Please find a path from the start (S) to the end (E) in the following maze. "
        "Answer format: [(x1,y1), (x2,y2), ..., (xn,yn)], each step is a coordinate, and each step must be adjacent and cannot go through walls. "
        "Note: x is column, y is row, i.e., (x,y) = (column,row). Please strictly follow this order for coordinates. The coordinate (0, 0) refers to the wall at the top-left corner, and the coordinate (width-1, length-1) refers to the wall at the bottom-right corner.\n"
        "If you believe the maze is unsolvable, please output \\boxed{unsolvable} at the end.\n"
        f"Maze:\n{maze_text}\nS is at (1,1), E is at (width-2,height-2)."
    )
    # data_source = "maze_train"
    item = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": full_prompt}],
        "ability": "alignment",
        "reward_model": {"style": "model", "ground_truth": solution_text},
        "extra_info": {"split": split, "index": idx}
    }
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

def load_mazes_from_text(filename="mazes.txt"):
    """
    Load mazes and their solutions from a plain text file.

    Args:
        filename (str): The name of the text file to load the data from.

    Returns:
        dict: A dictionary containing the mazes and their solutions.
    """
    # This function is left unimplemented as the user only requested saving in plain text format.
    pass

if __name__ == "__main__":
    # Example usage
    solvable_maze = [[0, 1, 0], [0, 1, 0], [0, 0, 0]]
    solvable_solution = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    unsolvable_maze = [[0, 1, 0], [0, 1, 1], [0, 0, 0]]
    obstacle_pos = (1, 2)

    save_solvable_maze_to_text(solvable_maze, solvable_solution)
    save_unsolvable_maze_to_text(unsolvable_maze, obstacle_pos)
    save_mazes_to_json_as_text(solvable_maze, solvable_solution, unsolvable_maze, obstacle_pos)
    print("Mazes saved to solvable_maze.txt, unsolvable_maze.txt, and mazes.json")