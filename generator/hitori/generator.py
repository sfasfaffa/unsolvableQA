import random
from collections import deque
from constraint import Problem
import json
import hashlib
import time
from tqdm import tqdm
# from .template import PROMPT_TEMPLATE
PROMPT_TEMPLATE = """
You are tasked with solving a Hitori puzzle.

### Rules:
1. The puzzle is played on an NxN grid, with each cell containing a number.
2. Your goal is to "black out" certain cells, following these rules:
   - In each row and column, the same number cannot appear more than once. To eliminate repetitions, you must black out some of the cells.
   - Black cells cannot be adjacent, either horizontally or vertically.
   - All white cells (cells that are not blacked out) must be connected, meaning you can travel between any two white cells through horizontal or vertical moves.

### Task:
Solve the following Hitori puzzle by blacking out the cells where needed.

You must provide the coordinates of the blacked-out cells.

### Response Format:
- Please output your answer within a code block (```), formatted as a list of coordinates, for example:
```
[(0, 0), (1, 3), (3, 2)]
```

Here is the puzzle: 
{question} """.strip()

PROMPT_TEMPLATE_ZH = """
你需要解决一个 Hitori 数独题目。

### 规则：
1. 该题目在一个 NxN 的网格上进行，每个格子里包含一个数字。
2. 你的目标是“涂黑”某些格子，遵循以下规则：
   - 每行和每列中，不能有相同的数字出现超过一次。为了消除重复，你必须涂黑一些格子。
   - 涂黑的格子不能相邻，不能水平或垂直相连。
   - 所有未涂黑的格子（即白格）必须是连通的，意味着你可以通过水平或垂直的方式，从一个白格到达另一个白格。

### 任务：
解决以下的 Hitori 数独题目，标出需要涂黑的格子。

你需要提供涂黑格子的坐标。

### 回答格式：
- 请在代码块内(```)输出你的答案，格式为坐标(r, c)列表，例如：
```
[(0, 0), (1, 3), (3, 2)]
```
请解决以下的题目：
{question}
""".strip()

# Board sizes used by the generator for each difficulty level. Note: these values
# determine N (the grid dimension) and may be odd or even. Earlier prompt text
# incorrectly stated that N must be even; in practice we use sizes from CONFIGS
# (e.g., 4, 5) and the templates should not assert 'N is even'.
CONFIGS = {
    "medium": 4,
    "hard": 5,
}

PUZZLE_TYPE = "search_puzzle"
SOURCE_URL = "auto-generated"
DATASET_NAME = "hitori"


def string_to_md5(s):
    encoded_string = s.encode('utf-8')
    md5_hash = hashlib.md5()
    md5_hash.update(encoded_string)
    return md5_hash.hexdigest()

def generate_random_board(n, min_val=1, max_val=None):
    """
    Generate a random n x n number board.
    By default, uses integers in the range 1..n, which can be adjusted as needed.
    """
    if max_val is None:
        max_val = n
    board = [[random.randint(min_val, max_val) for _ in range(n)] for _ in range(n)]
    return board

def is_connected_white(board, black_cells):
    """
    Determine if all white cells are connected given a black cell configuration.
    black_cells is a set of {(r,c), ...}.
    """
    n = len(board)
    # Find a white cell as the starting point for BFS
    start = None
    for r in range(n):
        for c in range(n):
            if (r, c) not in black_cells:
                start = (r, c)
                break
        if start:
            break
    
    if not start:
        # If there are no white cells, theoretically not a valid solution, can return False directly
        return False
    
    visited = set([start])
    queue = deque([start])
    dirs_ = [(1,0),(-1,0),(0,1),(0,-1)]
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in dirs_:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                if (nr, nc) not in black_cells and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    
    # Verify if all white cells are in visited
    for r in range(n):
        for c in range(n):
            if (r, c) not in black_cells:
                if (r, c) not in visited:
                    return False
    return True

def solve_hitori_with_constraint(board, stop_if_two=True):
    """
    Given a Hitori board (numbers only), use constraint solver to find all feasible black cell configurations.
    - When stop_if_two=True, stop early in the outer logic if 2 feasible solutions are found.
    
    Returns (number of feasible solutions, a set of black cells from one solution).
    If no solution, returns (0, None).
    If solution count >= 1, the second return value is one of the found solutions.
    """
    n = len(board)
    problem = Problem()
    
    # Define a binary variable x_{r,c} ∈ {0,1} for each cell
    # 0 = not black (white cell), 1 = black (black cell)
    all_cells = [(r,c) for r in range(n) for c in range(n)]
    for cell in all_cells:
        problem.addVariable(cell, [0, 1])

    # (A) Row/column number uniqueness constraint: if a number appears k times in a row/column => at least k-1 must be black
    # Rows
    for r in range(n):
        row_num_positions = {}
        for c in range(n):
            val = board[r][c]
            row_num_positions.setdefault(val, []).append((r,c))
        for val, positions in row_num_positions.items():
            if len(positions) > 1:
                k = len(positions)
                # sum(x_{r,c} in positions) >= k-1
                def make_sum_constraint(pos_list, k):
                    def _sum_constraint(*vals):
                        return sum(vals) >= k-1
                    return _sum_constraint
                problem.addConstraint(make_sum_constraint(positions, k), positions)
    
    # Columns
    for c in range(n):
        col_num_positions = {}
        for r in range(n):
            val = board[r][c]
            col_num_positions.setdefault(val, []).append((r,c))
        for val, positions in col_num_positions.items():
            if len(positions) > 1:
                k = len(positions)
                def make_sum_constraint(pos_list, k):
                    def _sum_constraint(*vals):
                        return sum(vals) >= k-1
                    return _sum_constraint
                problem.addConstraint(make_sum_constraint(positions, k), positions)

    # (B) Black cells cannot be adjacent
    for r in range(n):
        for c in range(n):
            if r + 1 < n:  
                problem.addConstraint(lambda a, b: a + b <= 1, ((r,c), (r+1,c)))
            if c + 1 < n:
                problem.addConstraint(lambda a, b: a + b <= 1, ((r,c), (r,c+1)))

    # (C) White cell connectivity (post-check)

    # If python-constraint version supports getSolutionIter() we can write:
    solutions_found = []
    get_solutions_iter = getattr(problem, "getSolutionIter", None)
    
    if get_solutions_iter is not None:
        # Version supports iterative solution retrieval, can manually break during enumeration
        for sol in problem.getSolutionIter():
            black_cells = {(r,c) for (r,c) in all_cells if sol[(r,c)] == 1}
            if is_connected_white(board, black_cells):
                solutions_found.append(black_cells)
                if stop_if_two and len(solutions_found) >= 2:
                    break
    else:
        # If getSolutionIter() is not available, get all solutions at once
        all_sol = problem.getSolutions()
        for sol in all_sol:
            black_cells = {(r,c) for (r,c) in all_cells if sol[(r,c)] == 1}
            if is_connected_white(board, black_cells):
                solutions_found.append(black_cells)
                if stop_if_two and len(solutions_found) >= 2:
                    break

    cnt = len(solutions_found)
    if cnt == 0:
        return 0, None
    else:
        # Return solution count and any one solution
        return cnt, solutions_found[0]

def generate_unique_hitori_problem(language, difficulty, max_tries=1000):
    n = CONFIGS.get(difficulty)
    """
    Generate n x n boards multiple times randomly and check uniqueness via the constraint solver.
    If a board with exactly one feasible solution is found within max_tries, return a dict with keys
    {"question": board, "answer": solution_list, "difficulty_level": difficulty}. Otherwise return None.
    """
    for _ in range(max_tries):
        board = generate_random_board(n, 1, n)
        sol_cnt, one_solution = solve_hitori_with_constraint(board, stop_if_two=True)
        if sol_cnt == 1:
            solution_list = list(one_solution)
            return {
                "question":board,
                "answer": solution_list,
                "difficulty_level": difficulty
            }
    return None


def transform_problem_to_meta(problem, idx, language, split):
    timestamp = str(time.time())
    random_suffix = random.randint(0, int(1e6))
    id_string = f"binario_{idx}_{timestamp}_{random_suffix}"
    hash_id_string = string_to_md5(id_string)
    return {
        "id": hash_id_string,
        "question": problem["question"],
        "answer": problem["answer"],
        "rationale": "",
        "split": split,
        "type": PUZZLE_TYPE,
        "source_url": SOURCE_URL,
        "dataset_name": DATASET_NAME,
        "difficulty_level": problem["difficulty_level"],
        "language": language,
    }

def generate(count=100, difficulty="medium", language="en", split="train", force_solvable=None):
    prompt_template = PROMPT_TEMPLATE
    generated = 0
    attempts = 0
    max_attempts = count * 10

    while generated < count and attempts < max_attempts:
        try:
            # inner sampling loop (bounded) to find a suitable problem
            tries = 0
            problem = None
            while tries <= 2000:
                if force_solvable is False:
                    # Need an unsolvable board: sample and check directly
                    b = generate_random_board(CONFIGS.get(difficulty), 1, CONFIGS.get(difficulty))
                    sol_cnt, _ = solve_hitori_with_constraint(b, stop_if_two=False)
                    if sol_cnt == 0:
                        problem = {"question": b, "answer": "unsolvable", "difficulty_level": difficulty}
                        break
                else:
                    # Prefer boards with a unique solution (force_solvable True or None)
                    problem = generate_unique_hitori_problem(language, difficulty, max_tries=1000)
                    if problem:
                        break

                tries += 1

            # If inner attempts failed to produce a problem, increment attempts and retry outer loop
            if not problem:
                print(f"    Warning: failed to generate a suitable problem for difficulty={difficulty} in inner sampling (tries={tries}); retrying outer loop.")
                attempts += 1
                continue

            meta = transform_problem_to_meta(problem, generated, language, split)
            meta["task_name"] = DATASET_NAME
            yield {
                "prompt": prompt_template.format(question=meta["question"]),
                "answer": meta["answer"] if isinstance(meta["answer"], list) else ("unsolvable" if str(meta.get("answer")).strip() in ["unsolvable"] else meta["answer"]),
                "task_name": DATASET_NAME,
                "ability": PUZZLE_TYPE,
                "language": language,
                "meta": json.dumps(meta, ensure_ascii=False),
            }
            generated += 1
        except ValueError as e:
            print(f"Generation error: {e}")
        attempts += 1

    if attempts >= max_attempts:
        print(f"Warning: Maximum attempt count reached, generated {generated} / {count} puzzles.")

def save_to_jsonl(output_file, count, language, split, force_solvable=None):
    difficulties = ["medium", "hard"]
    per_difficulty = count // len(difficulties)
    remainder = count % len(difficulties)
    difficulty_counts = {d: per_difficulty for d in difficulties}
    for i in range(remainder):
        difficulty_counts[difficulties[i]] += 1

    entries = []
    with open(output_file, 'w', encoding='utf-8') as f:
        for difficulty in difficulties:
            num = difficulty_counts[difficulty]
            if num == 0:
                continue
            print(f"Generating {difficulty} puzzles: {num} puzzles")
            for item in tqdm(generate(num, difficulty=difficulty, language=language, split=split, force_solvable=force_solvable), desc=f"Generating {difficulty} puzzles"):
                meta = json.loads(item['meta']) if isinstance(item.get('meta'), str) else item.get('meta')
                is_unsolvable = (isinstance(item.get('answer'), str) and item.get('answer').strip() in ["unsolvable", "No valid solution exists for the given Hitori puzzle."])
                data_source = f"hitori_unsolvable" if is_unsolvable else "hitori"
                prompt_content = item['prompt'] + ("\nIf there is no valid solution, please output \\boxed{unsolvable} at the end.\n")
                entry = {
                    "data_source": data_source,
                    "prompt": [{"content": prompt_content, "role": "user"}],
                    "ability": item.get('ability', PUZZLE_TYPE),
                    "reward_model": {"ground_truth": "unsolvable" if is_unsolvable else item.get('answer'), "style": "unsolvable_generated" if is_unsolvable else "generated"},
                    "extra_info": {"index": meta.get('id', f"idx_{random.randint(0, int(1e9))}"), "question": meta.get('question')}
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                entries.append(entry)
    try:
        import pandas as pd
        pd.DataFrame(entries).to_parquet(output_file.rsplit('.',1)[0] + '.parquet', index=False)
    except Exception as e:
        print(f"Warning: failed to write parquet for {output_file}: {e}")

if __name__ == "__main__":
    #save_to_jsonl("train_en.jsonl", 20000, language="en", split="train")
    save_to_jsonl("train_en_hitori.jsonl", 50, language="en", split="eval", force_solvable=True)
    save_to_jsonl('train_en_hitori_unsolvable.jsonl', 50, language='en', split="eval", force_solvable=False)
    save_to_jsonl("test_en_hitori.jsonl", 50, language="en", split="eval", force_solvable=True)
    save_to_jsonl('test_en_hitori_unsolvable.jsonl', 50, language='en', split="eval", force_solvable=False)

