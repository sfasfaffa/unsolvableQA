import json
import pandas as pd
from maze_unsolvable_constructor import construct_unsolvable_maze, print_maze
from maze_generator import generate_maze
from maze_storage import save_mazes_to_json_as_text, save_maze_to_jsonl_format
import random

def maze_to_text(maze):
    return "\n".join("".join(str(cell) for cell in row) for row in maze)

def solution_to_text(solution):
    return " -> ".join(f"({x},{y})" for x, y in solution)

def main():
    width, height, difficulty = 8, 8, 7
    # Generate solvable maze
    solvable_maze, solvable_solution = generate_maze(width, height)
    print("Solvable maze:")
    print_maze(solvable_maze, solvable_solution)

    # Save solvable maze to JSON file as text
    save_mazes_to_json_as_text(solvable_maze, solvable_solution, [], None, filename="solvable_maze.json")

    # Generate unsolvable maze
    maze_unsolvable, obstacle_pos, _, _ = construct_unsolvable_maze(width, height, difficulty)
    if maze_unsolvable:
        print("\nUnsolvable maze:")
        print_maze(maze_unsolvable)
        print(f"Obstacle at {obstacle_pos}")

        # Save unsolvable maze to JSON file as text
        save_mazes_to_json_as_text([], [], maze_unsolvable, obstacle_pos, filename="unsolvable_maze.json")
    else:
        print("\nFailed to generate an unsolvable maze.")

    jsonl_filename = "mazes_test_hard.jsonl"
    parquet_filename = "mazes_test_hard.parquet"
    records = []
    seen = set()
    idx = 0
    attempts_total = 0
    while idx < 100 and attempts_total < 1000:
        attempts_total += 1
        width = random.randint(9, 12)
        height = random.randint(9, 12)
        # 可解迷宫
        # use random seed for more variability
        maze, solution = generate_maze(width, height)
        maze_text = maze_to_text(maze)
        # skip duplicate maze layout
        if maze_text in seen:
            continue
        seen.add(maze_text)
        solution_text = solution_to_text(solution)
        data_source = "maze_test_hard"
        split = "test"
        full_prompt = (
            "Please find a path from the start (S) to the end (E) in the following maze. "
            "Answer format: [(x1,y1), (x2,y2), ..., (xn,yn)], each step is a coordinate, and each step must be adjacent and cannot go through walls. "
            "Note: x is column, y is row, i.e., (x,y) = (column,row). Please strictly follow this order for coordinates. The coordinate (0, 0) refers to the wall at the top-left corner, and the coordinate (width-1, length-1) refers to the wall at the bottom-right corner.\n"
            "If you believe the maze is unsolvable, please output \\boxed{unsolvable} at the end.\n"
            f"Maze:\n{maze_text}\nS is at (1,1), E is at (width-2,height-2)."
        )
        save_maze_to_jsonl_format(data_source, maze_text, solution_text, split, idx, filename=jsonl_filename)
        record = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": full_prompt}],
            "ability": "alignment",
            "reward_model": {"style": "model", "ground_truth": solution_text},
            "extra_info": {"split": split, "index": idx}
        }
        records.append(record)
        # Attempt to generate an unsolvable maze (unique layout). If duplicate, skip.
        maze_unsolvable, obstacle_pos, _, _ = construct_unsolvable_maze(width, height, difficulty=7)
        if maze_unsolvable:
            maze_unsolvable_text = maze_to_text(maze_unsolvable)
            if maze_unsolvable_text not in seen:
                seen.add(maze_unsolvable_text)
                data_source_unsolvable = "maze_test_hard_unsolvable"
                full_prompt_unsolvable = (
                    "Please find a path from the start (S) to the end (E) in the following maze. "
                    "Answer format: [(x1,y1), (x2,y2), ..., (xn,yn)], each step is a coordinate, and each step must be adjacent and cannot go through walls. "
                    "Note: x is column, y is row, i.e., (x,y) = (column,row). Please strictly follow this order for coordinates. The coordinate (0, 0) refers to the wall at the top-left corner, and the coordinate (width-1, length-1) refers to the wall at the bottom-right corner.\n"
                    "If you believe the maze is unsolvable, please output \\boxed{unsolvable} at the end.\n"
                    f"Maze:\n{maze_unsolvable_text}\nS is at (1,1), E is at (width-2,height-2)."
                )
                save_maze_to_jsonl_format(data_source_unsolvable, maze_unsolvable_text, "No solution", split, idx, filename=jsonl_filename)
                record_unsolvable = {
                    "data_source": data_source_unsolvable,
                    "prompt": [{"role": "user", "content": full_prompt_unsolvable}],
                    "ability": "alignment",
                    "reward_model": {"style": "model", "ground_truth": "No solution"},
                    "extra_info": {"split": split, "index": idx}
                }
                records.append(record_unsolvable)
            else:
                # duplicate unsolvable layout; skip
                pass
        # increment idx only when at least one record was added for this index
        idx += 1
    # Save to Parquet
    # records#随机打乱
    random.shuffle(records)
    df = pd.DataFrame(records)
    df.to_parquet(parquet_filename, engine="pyarrow")
    print(f"已保存100个可解和不可解迷宫到 {jsonl_filename} 和 {parquet_filename}")

if __name__ == "__main__":
    main()
