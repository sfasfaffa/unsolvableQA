import random
import collections
import json
from tqdm import tqdm
# from .template import PROMPT_TEMPLATE
PROMPT_TEMPLATE = """The 8 Puzzle is a classic sliding puzzle game. It consists of a 3×3 grid containing 8 numbered tiles (1–8) and one blank space (0). The goal is to arrange tiles in order.

Rules Summary:
- Move only tiles adjacent to the blank (0).
- Allowed moves: up (U), down (D), left (L), right (R).
- Move semantics (intuitive):
    - 'U' means the blank (0) swaps with the tile ABOVE it (the blank moves up).
    - 'D' means the blank (0) swaps with the tile BELOW it (the blank moves down).
    - 'L' means the blank (0) swaps with the tile to its LEFT (the blank moves left).
    - 'R' means the blank (0) swaps with the tile to its RIGHT (the blank moves right).
- Goal state:

1  2  3
4  5  6
7  8  0

Quick move demo (before → after):
U:
1  2  3   1  0  3
4  0  6 → 4  2  6
7  5  8   7  5  8

D:
1  2  3   1  2  3
4  0  6 → 4  5  6
7  5  8   7  0  8

L:
1  2  3   1  2  3
4  0  6 → 0  4  6
7  5  8   7  5  8

R:
1  2  3   1  2  3
4  0  6 → 4  6  0
7  5  8   7  5  8

Response Format (strict):
- Return your answer inside a triple-backtick code block (```...```).
- Inside the code block, output ONLY ONE of:
        1) A move sequence: uppercase letters without spaces over {{L,R,U,D}} (e.g., `LRURDL`)
        2) If no solution exists: exactly `unsolvable`
- Do not include any extra words or punctuation inside the code block.

- If there is an answer, the <result> is the sequence of moves, for example:
```
LRURDL
```

Question:
<your question>
{question}
</your question>
Return your final answer strictly in the required code-block format.
""".strip()
def diff_j(board, K):
    # Flatten the board into a one-dimensional list
    flat_board = [tile for row in board for tile in row if tile != 0]
    inversions = 0

    # Calculate the number of inversions
    for i in range(len(flat_board)):
        for j in range(i + 1, len(flat_board)):
            if flat_board[i] > flat_board[j]:
                inversions += 1

    return inversions

def is_solvable(board, K):
    """Solvability check.
    - For K odd (e.g., 3x3): solvable iff number of inversions (excluding 0) is even.
    - For K even: depends on blank row parity from bottom combined with inversion parity.
    """
    flat_board = [tile for row in board for tile in row if tile != 0]
    inversions = 0
    for i in range(len(flat_board)):
        for j in range(i + 1, len(flat_board)):
            if flat_board[i] > flat_board[j]:
                inversions += 1

    if K % 2 == 1:
        return inversions % 2 == 0
    # K even: compute blank row from bottom (row index starting at 1)
    blank_r, blank_c = next(((ri, ci) for ri, row in enumerate(board) for ci, v in enumerate(row) if v == 0), (None, None))
    if blank_r is None:
        return False
    blank_row_from_bottom = K - blank_r
    # Standard rule: solvable if (blank_row_from_bottom is odd and inversions even)
    # or (blank_row_from_bottom is even and inversions odd). Equivalent to parity sum being odd.
    return (inversions + blank_row_from_bottom) % 2 == 1

# Generate a random K-puzzle board
def generate_puzzle(K, diff):
    # Create a list from 0 to K^2-1, where 0 represents the empty space
    tiles = list(range(K**2))
    random.shuffle(tiles)

    # Convert the list to a KxK board
    board = [tiles[i:i + K] for i in range(0, K**2, K)]
    inv = diff_j(board, K)
    #print(inv, diff)
    # If the board is not solvable or doesn't meet difficulty requirements, regenerate
    while (not is_solvable(board, K)) or (not diff[0] <= inv <= diff[1]):
        random.shuffle(tiles)
        board = [tiles[i:i + K] for i in range(0, K**2, K)]
        inv = diff_j(board, K)

    return [board, inv]


def board_to_string(board):
    board_str = '\n'.join(' '.join(str(tile).rjust(2, ' ') for tile in row) for row in board)
    return board_str

def solve_puzzle_bfs(board):
    """Return a shortest move sequence over 'LRUD' to reach goal, or None if unsolvable.
    Goal state is:
    1 2 3
    4 5 6
    7 8 0
    """
    K = len(board)
    # Flatten helpers
    def to_tuple(b):
        return tuple(x for row in b for x in row)
    def from_tuple(t):
        return [list(t[i*K:(i+1)*K]) for i in range(K)]
    start = to_tuple(board)
    goal = tuple(list(range(1, K*K)) + [0])
    if start == goal:
        return ""  # already solved
    # Locate zero in tuple index form
    def zero_pos(t):
        i = t.index(0)
        return divmod(i, K)
    # Moves: U,D,L,R where semantics are blank swaps with neighbor
    moves = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1),
    }
    from collections import deque
    q = deque()
    q.append((start, ""))
    seen = {start}
    while q:
        state, path = q.popleft()
        zr, zc = zero_pos(state)
        for m, (dr, dc) in moves.items():
            nr, nc = zr + dr, zc + dc
            if 0 <= nr < K and 0 <= nc < K:
                # swap blank with target
                idx_blank = zr*K + zc
                idx_target = nr*K + nc
                lst = list(state)
                lst[idx_blank], lst[idx_target] = lst[idx_target], lst[idx_blank]
                ns = tuple(lst)
                if ns in seen:
                    continue
                if ns == goal:
                    return path + m
                seen.add(ns)
                q.append((ns, path + m))
    return None

def generate_easy_by_scramble(K=3, steps=4):
    """Generate a solvable board by scrambling from the goal with random moves.
    Constraints: 4-8 steps, no immediate backtrack (no consecutive opposite moves).
    Returns (board, sequence) where sequence solves the scrambled board.
    """
    assert 4 <= steps <= 8
    # Goal state
    goal = [[(r*K + c + 1) % (K*K) for c in range(K)] for r in range(K)]
    # Move definitions
    moves = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1),
    }
    opposite = {'U':'D','D':'U','L':'R','R':'L'}
    # Find blank
    def find_zero(b):
        for r in range(K):
            for c in range(K):
                if b[r][c] == 0:
                    return r, c
        return None, None
    b = [row[:] for row in goal]
    zr, zc = find_zero(b)
    scramble_moves = []
    prev = None
    import random as _rand
    for _ in range(steps):
        cand = []
        for m, (dr, dc) in moves.items():
            if prev and m == opposite.get(prev):
                continue  # prevent immediate backtrack
            nr, nc = zr + dr, zc + dc
            if 0 <= nr < K and 0 <= nc < K:
                cand.append((m, dr, dc))
        if not cand:
            # fallback allow any move
            for m, (dr, dc) in moves.items():
                nr, nc = zr + dr, zc + dc
                if 0 <= nr < K and 0 <= nc < K:
                    cand.append((m, dr, dc))
        m, dr, dc = _rand.choice(cand)
        nr, nc = zr + dr, zc + dc
        b[zr][zc], b[nr][nc] = b[nr][nc], b[zr][zc]
        scramble_moves.append(m)
        prev = m
        zr, zc = nr, nc
    # The solution is the inverse of scramble
    inverse = {'U':'D','D':'U','L':'R','R':'L'}
    solution = ''.join(inverse[m] for m in reversed(scramble_moves))
    return b, solution

def generate(count=100, difficulty='medium', language='en', split="train", force_solvable=None, steps_range=None):
    prompt_template = PROMPT_TEMPLATE
    exist = {}
    dif_level = {"easy" : [0, 10], "medium" : [12,16], "hard" : [18,100]}
    diff = dif_level[difficulty]
    K = 3
    generated = 0  # Track actual generated count
    attempts_total = 0
    max_total_attempts = count * 100  # Prevent infinite loop
    
    pbar = tqdm(total=count)
    while generated < count and attempts_total < max_total_attempts:
        attempts_total += 1
        # Sample a puzzle and enforce solvability/unsolvability if requested
        attempts = 0
        max_attempts = 5000
        def make_unsolvable(board, K):
            # Deterministically flip parity to create an unsolvable instance.
            # Strategy:
            # - For odd K: swap two non-zero tiles to flip inversion parity.
            # - For even K: if parity sum (inversions + blank_row_from_bottom) is odd (solvable),
            #   swap two non-zero tiles to toggle inversions parity, making sum even (unsolvable).
            b = [row[:] for row in board]
            # find positions of two non-zero different tiles
            pos = []
            for r in range(K):
                for c in range(K):
                    if b[r][c] != 0:
                        pos.append((r, c))
            if len(pos) >= 2:
                (r1, c1), (r2, c2) = pos[0], pos[1]
                b[r1][c1], b[r2][c2] = b[r2][c2], b[r1][c1]
            return b

        while True:
            # If we target short solutions, generate by scrambling from goal
            if force_solvable is True and steps_range:
                Lmin, Lmax = steps_range
                steps = random.randint(max(4, Lmin), min(8, Lmax))
                board, seq_try = generate_easy_by_scramble(K, steps)
                inv = diff_j(board, K)
                solv = True
            else:
                board, inv = generate_puzzle(K, diff)
                solv = is_solvable(board, K)
            if force_solvable is True:
                if solv:
                    break
                else:
                    attempts += 1
            elif force_solvable is False:
                if not solv:
                    break
                else:
                    # deterministically convert to unsolvable by parity flip
                    board = make_unsolvable(board, K)
                    inv = diff_j(board, K)
                    solv = is_solvable(board, K)
                    if not solv:
                        break
                    attempts += 1
            else:
                break
            if attempts >= max_attempts:
                # As a final fallback, randomize and apply parity flip when unsolvable is required
                tiles = list(range(K**2))
                random.shuffle(tiles)
                board = [tiles[i:i + K] for i in range(0, K**2, K)]
                if force_solvable is False:
                    board = make_unsolvable(board, K)
                inv = diff_j(board, K)
                break
        board_str = board_to_string(board)
        is_solv = is_solvable(board, K)
        
        # CRITICAL: Enforce strict solvability requirements based on force_solvable flag
        if force_solvable is True and not is_solv:
            # FATAL: Requested solvable but got unsolvable - skip this item
            continue
        if force_solvable is False and is_solv:
            # FATAL: Requested unsolvable but got solvable - skip this item
            continue
        
        # Produce answer matching prompt: sequence for solvable, 'unsolvable' otherwise
        if is_solv:
            # If we created an easy scramble, we may already have a sequence
            if steps_range and force_solvable:
                # recompute shortest to be safe, but prefer scramble inverse
                seq_scramble = None
                try:
                    # try recover by solving; if same length within range, keep
                    seq_bfs = solve_puzzle_bfs(board)
                    seq = seq_bfs if seq_bfs is not None else None
                except Exception:
                    seq = None
                if seq is None:
                    # As a fallback, compute BFS again
                    seq = solve_puzzle_bfs(board)
            else:
                seq = solve_puzzle_bfs(board)
            # Fallback safety: if solver failed, skip this sample
            if seq is None:
                continue
            answer_str = seq
        else:
            answer_str = "unsolvable"
        
        # DOUBLE CHECK: Final verification before yielding
        assert (force_solvable is None) or (force_solvable is True and is_solv) or (force_solvable is False and not is_solv), \
            f"Solvability mismatch: force_solvable={force_solvable}, is_solvable={is_solv}"
        
        yield {
            "prompt": prompt_template.format(question=board_str),
            "answer":  answer_str,
            "task_name": "eight_puzzle",    
            "ability": "logic_puzzle", 
            "language": language,
            "meta": json.dumps({
                "id":"8-puzzle_"+difficulty+str(generated),
                "question": board,
                "answer": answer_str,
                "inversion": inv,
                "rationale": "", 
                "split": split,
                "type": "sequential_puzzle", 
                "source_url": "auto-generated", 
                "dataset_name": "eight_puzzle", 
                "difficulty_level": difficulty,
                "language": language,
            })
        }
        generated += 1
        pbar.update(1)
    
    pbar.close()
    if generated < count:
        print(f"Warning: Only generated {generated}/{count} items after {attempts_total} attempts")

def save_to_jsonl(of_jsonl, of_meta, count, lange='en', force_solvable=None, easy_steps_ratio=0.5, easy_steps_range=(3,7)):
    """Save standardized entries with canonical unsolvable handling.
    Mirrors game24's save_to_jsonl: writes problems to of_jsonl and metas to of_meta.
    """
    import os
    for path in (of_jsonl, of_meta):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    entries = []
    seen_keys = set()
    with open(of_jsonl, 'w', encoding='utf-8') as f1, open(of_meta, 'w', encoding='utf-8') as f2:
        # distribute evenly across difficulties
        per = count // 3
        rem = count % 3
        diffs = ['easy', 'medium', 'hard']
        counts = {d: per for d in diffs}
        for d in diffs:
            if rem <= 0:
                break
            counts[d] += 1
            rem -= 1

        for difficulty in diffs:
            num = counts[difficulty]
            if num <= 0:
                continue
            # Split into easy-steps solvable subset and normal subset when generating solvable data
            if force_solvable is True and num > 1 and easy_steps_ratio > 0:
                num_easy = max(1, int(num * easy_steps_ratio))
                num_normal = num - num_easy
            else:
                num_easy = 0
                num_normal = num

            # Easy subset: enforce BFS path length in easy_steps_range
            if num_easy > 0:
                for item in generate(num_easy, difficulty, lange, split="train", force_solvable=True, steps_range=easy_steps_range):
                    meta = json.loads(item['meta']) if isinstance(item.get('meta'), str) else item.get('meta')
                    is_unsolvable = (isinstance(item.get('answer'), str) and item.get('answer').strip() == 'unsolvable')
                    data_source = 'eight_puzzle_unsolvable' if is_unsolvable else 'eight_puzzle'
                    prompt_content = item['prompt']
                    # Dedup key based on prompt content + answer target
                    key = json.dumps({"prompt": prompt_content, "answer": item.get('answer')}, ensure_ascii=False)
                    if key in seen_keys:
                        continue
                    entry = {
                        "data_source": data_source,
                        "prompt": [{"content": prompt_content, "role": "user"}],
                        "ability": item.get('ability', 'logic_puzzle'),
                        "reward_model": {"ground_truth": "unsolvable" if is_unsolvable else item.get('answer'), "style": "unsolvable_generated" if is_unsolvable else "generated"},
                        "extra_info": {"index": meta.get('id', f"idx_{random.randint(0, int(1e9))}"), "question": meta.get('question')}
                    }
                    f1.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    f2.write(json.dumps(meta, ensure_ascii=False) + '\n')
                    entries.append(entry)
                    seen_keys.add(key)

            # Normal subset: existing behavior
            for item in generate(num_normal, difficulty, lange, split="train", force_solvable=force_solvable):
                meta = json.loads(item['meta']) if isinstance(item.get('meta'), str) else item.get('meta')
                is_unsolvable = (isinstance(item.get('answer'), str) and item.get('answer').strip() == 'unsolvable')
                data_source = 'eight_puzzle_unsolvable' if is_unsolvable else 'eight_puzzle'
                prompt_content = item['prompt']
                key = json.dumps({"prompt": prompt_content, "answer": item.get('answer')}, ensure_ascii=False)
                if key in seen_keys:
                    continue
                entry = {
                    "data_source": data_source,
                    "prompt": [{"content": prompt_content, "role": "user"}],
                    "ability": item.get('ability', 'logic_puzzle'),
                    "reward_model": {"ground_truth": "unsolvable" if is_unsolvable else item.get('answer'), "style": "unsolvable_generated" if is_unsolvable else "generated"},
                    "extra_info": {"index": meta.get('id', f"idx_{random.randint(0, int(1e9))}"), "question": meta.get('question')}
                }
                f1.write(json.dumps(entry, ensure_ascii=False) + '\n')
                f2.write(json.dumps(meta, ensure_ascii=False) + '\n')
                entries.append(entry)
                seen_keys.add(key)

    # optional parquet alongside jsonl
    try:
        import pandas as pd
        parquet_path = of_jsonl.rsplit('.', 1)[0] + '.parquet'
        pd.DataFrame(entries).to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f"Warning: failed to write parquet for {of_jsonl}: {e}")

# NOTE: Removed duplicate, legacy save_to_jsonl to avoid overriding the standardized saver above.

# Call functions to generate and save
#save_to_jsonl('training/8puzzle/en/train.jsonl', 'raw/8puzzle/en/train.jsonl', 24000, 'en')
#save_to_jsonl('training/8puzzle/zh/train.jsonl', 'raw/8puzzle/zh/train.jsonl',24000, 'zh')

#save_to_jsonl('eval/8puzzle/en/test.jsonl', 'raw/8puzzle/en/test.jsonl', 1500, 'en')
#save_to_jsonl('eval/8puzzle/zh/test.jsonl', 'raw/8puzzle/zh/test.jsonl',1500, 'zh')

if __name__ == "__main__":
    # Mirror game24 main: produce solvable and unsolvable sets
    try:
        # Small sample for quick verification; adjust counts as needed
        save_to_jsonl('test_en_eight_puzzle.jsonl', 'test_en_eight_puzzle_meta.jsonl', 100, 'en', force_solvable=True)
        save_to_jsonl('test_en_eight_puzzle_unsolvable.jsonl', 'test_en_eight_puzzle_meta_unsolvable.jsonl', 100, 'en', force_solvable=False)
        print('Eight-puzzle datasets generated successfully.')
    except Exception as e:
        print('Error generating eight-puzzle datasets:', e)
