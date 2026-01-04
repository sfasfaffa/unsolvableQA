import random
import collections
import time
from typing import Literal, List, Set, Callable, Tuple, Optional
import json
from tqdm import tqdm
import hashlib
import re
import os
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Templates & Configuration
# -----------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are tasked with solving a grid puzzle. This type of puzzle requires careful analysis of the provided background information and clues to deduce the correct arrangement of elements in a grid format. Follow the steps below to solve the puzzle and present your solution in the specified format.

1. **Background Information:** Carefully read any introductory information provided with the puzzle. This may include context or specific constraints that apply to the puzzle.

2. **Clues:** Analyze each clue given. These clues will guide you in determining the relationships between different elements in the grid.

3. **Logical Deduction:** Use logical reasoning to deduce the correct placement of each element in the grid. Consider all possible options and eliminate those that contradict the clues.

4. **Consistency Check:** Ensure that your solution is consistent with all the clues and background information provided.

Your response should include a solution followed by the final answer in a markdown table format. Use the following structure:

Assume the columns are: {columns_desc}.

Final Answer:
{table_header}
| Value 1 | [Correct Value] | [Correct Value] | ...
...

Here is the puzzle:

{question}

You must stick to the given uncompleted table and must not transpose the table.
""".strip()

SOURCE_URL = "auto-generated"
PUZZLE_TYPE = "grid_puzzle"
DATASET_NAME = "zebra_logic"
ROWS = 4 

CONFIGS = {
    "easy": {"level": 2, "columns": 3, "rows": ROWS},
    "medium": {"level": 8, "columns": 4, "rows": ROWS},
    "hard": {"level": 15, "columns": 5, "rows": ROWS},
}

# -----------------------------------------------------------------------------
# 2. Core Logic Puzzle Generator Engine
# -----------------------------------------------------------------------------

def format_table_no_header(header: List[str], table: List[List[str]],
                 top_format='{:^{}}', left_format=' {:<{}}', cell_format='{:<{}}',
                 col_delim=' | ', row_delim='\n', prefix_format='|', postfix_format='|'):
    table_format = len(table) * [[prefix_format + left_format] + len(header) * [cell_format]]
    col_widths = [max(len(format.format(cell, 0))
                      for format, cell in zip(col_format, col))
                  for col_format, col in zip(zip(*table_format), zip(*table))]
    return row_delim.join(
               col_delim.join(
                   format.format(cell, width)
                   for format, cell, width in zip(row_format, row, col_widths)) + f" {postfix_format}"
               for row_format, row in zip(table_format, table))


def update_range(wns: List[str], rns: List[List[Set[str]]], cmp: Callable):
    changed = False
    for rn in rns:
        classified_words = set()
        for n_col, set_of_words in enumerate(rn):
            if len(set_of_words) == 1:
                classified_words.add(next(iter(set_of_words)))
        word_to_cols = dict()
        for n_col, set_of_words in enumerate(rn):
            if len(set_of_words) != 1:
                prev_length = len(set_of_words)
                set_of_words.difference_update(classified_words)
                changed |= prev_length != len(set_of_words)
                for word in set_of_words:
                    word_to_cols.setdefault(word, set()).add(n_col)
        for word, cols in word_to_cols.items():
            if len(cols) == 1:
                x = rn[next(iter(cols))]
                if len(x) != 1:
                    x.clear()
                    x.add(word)
                    changed = True

    new_rns = [[{x for x in xs if x != wn} for xs in rn] for wn, rn in zip(wns, rns)]
    pairs = []
    for wn, rn in zip(wns, rns):
        new_pairs = []
        break_condition = True
        for cn, setn in enumerate(rn):
            if wn in setn:
                break_condition = False
                if not pairs:
                    pairs = [[]]
                for v in pairs:
                    new_pairs.append([*v, cn])
        pairs = new_pairs
        if break_condition:
            break
    for pair in pairs:
        if cmp(*pair):
            for nrn, cn, wn in zip(new_rns, pair, wns):
                nrn[cn].add(wn)
    changed |= any(rn != new_rn for rn, new_rn in zip(rns, new_rns))
    if changed:
        for rn, new_rn in zip(rns, new_rns):
            for old, new in zip(rn, new_rn):
                old.intersection_update(new)
    return changed


def update_ranges(relations, ranges):
    changed = False
    for ins, wns, callable_object, *_ in relations:
        changed |= update_range(wns, [ranges[i] for i in ins], callable_object)
    return changed


def generate_puzzle_logic(table: List[List[str]], *,
                    level: int,
                    minimal_conditions: bool = False, max_seconds_for_minimizing: float = None,
                    tries: int = 10):
    """Core logic to generate clues from a table."""
    if level not in range(1, 20 + 1):
        level = 10 

    table_wo_left = [row[1:] for row in table]
    n_attributes = len(table_wo_left)
    m_objects = len(table_wo_left[0])

    center = m_objects // 2
    except_flag = True
    
    rules_for_relations = [
        (2, lambda j1, j2: j1 == j2, ['{0}:{1} == {2}:{3}', '{2}:{3} == {0}:{1}']),
        (2, lambda j1, j2: j1 == j2 - 1, ['{0}:{1} is on the left of {2}:{3}']),
        (2, lambda j1, j2: j1 == j2 + 1, ['{0}:{1} is on the right of {2}:{3}']),
        (1, lambda j1: j1 == 0, ['{0}:{1} is on the far left']),
        (1, lambda j1, last_index=m_objects - 1: j1 == last_index, ['{0}:{1} is on the far right']),
    ] + (m_objects % 2 != 0) * [(1, lambda j1, mid=center: j1 == mid, ['{0}:{1} is in the middle'])]
    
    if level >= 2:
        rules_for_relations += [
            (3, lambda j1, j2, j3: j2 + 1 == j1 == j3 - 1 or j3 + 1 == j1 == j2 - 1,
             ['{0}:{1} is between {2}:{3} and {4}:{5}', '{0}:{1} is between {4}:{5} and {2}:{3}']),
        ]
    if level >= 3:
        rules_for_relations += [
            (2, lambda j1, j2: j1 == j2 - 1 or j1 == j2 + 1,
             ['{0}:{1} is on the left or right of {2}:{3}']),
            (1, lambda j1, last_index=m_objects - 1: j1 == 0 or j1 == last_index,
             ['{0}:{1} is on the far left or far right']),
        ]
    if level >= 4:
        rules_for_relations += [
            (1, lambda j1: (j1 + 1) % 2 != 0, ['{0}:{1} is in an odd position']),
            (1, lambda j1: (j1 + 1) % 2 == 0, ['{0}:{1} is in an even position']),
        ]
    if level >= 5:
        rules_for_relations += [
            (2, lambda j1, j2: j1 < j2, ['{0}:{1} is somewhere to the left of {2}:{3}']),
            (2, lambda j1, j2: j1 > j2, ['{0}:{1} is somewhere to the right of {2}:{3}']),
        ]
    if level >= 6:
         rules_for_relations += [(2, lambda j1, j2: j1 != j2, ['{0}:{1} != {2}:{3}'], except_flag)]

    # Main generation loop
    min_relations = None
    max_total_loops = 1000 
    current_loop = 0
    
    while True:
        current_loop += 1
        if current_loop > max_total_loops:
            return [] 

        ranges = [[set(table_wo_left[i]) for _ in range(len(table_wo_left[i]))] for i in range(len(table_wo_left))]
        relations = list()
        fail = False
        while not fail:
            needs_clarification = list()
            no_solutions = False
            solved = True
            for i, rng in enumerate(ranges):
                for j, rs in enumerate(rng):
                    if len(rs) == 0:
                        no_solutions = True
                        solved = False
                        break
                    elif len(rs) > 1:
                        solved = False
                        needs_clarification.append((i, j))
                if no_solutions:
                    break
            
            if solved or (min_relations is not None and len(relations) >= len(min_relations)):
                tries -= 1
                if min_relations is None or len(relations) < len(min_relations):
                    min_relations = relations
                if tries > 0:
                    fail = True
                    continue
            
            if tries <= 0:
                return [t[-1] for t in (min_relations if min_relations else relations)]

            if no_solutions or not needs_clarification:
                fail = True
                continue

            i, j = item = random.choice(needs_clarification)
            next_i = random.randint(0, n_attributes - 1)
            next_j = random.randint(0, m_objects - 1)
            
            valid_rule_found = False
            random.shuffle(rules_for_relations)
            
            for rule in rules_for_relations:
                n_args, cmp_func, strs, *flags = rule
                if n_args == 2:
                    if cmp_func(j, next_j):
                        str_fmt = random.choice(strs)
                        val1 = table_wo_left[i][j]
                        val2 = table_wo_left[next_i][next_j]
                        attr1 = table[i][0]
                        attr2 = table[next_i][0]
                        
                        relations.append(([i, next_i], [val1, val2], cmp_func, str_fmt.format(attr1, val1, attr2, val2)))
                        valid_rule_found = True
                        break
            
            if valid_rule_found:
                changed = True
                while changed:
                    changed = update_ranges(relations, ranges)
            else:
                fail = True

        if not fail:
            break

    return [t[-1] for t in relations]

def get_empty_table(table):
    empty_table = []
    for row in table:
        empty_table.append([row[0]] + ["..." for i in range(1, len(row))])
    return empty_table

def generator_zebra_logic_problem(rows=4, columns=5, level=18):
    kinds_dict = {
        "Nationality": ["american", "british", "canadian", "chinese", "dutch", "french", "german", "italian", "japanese", "mexican"],
        "Food": ["apple", "banana", "cake", "donut", "eggs", "fries", "grapes", "hamburger", "ice-cream", "jelly"],
        "Pet": ["bird", "cat", "dog", "fish", "horse", "lion", "mouse", "rabbit", "snake", "tiger"],
        "Job": ["artist", "baker", "chef", "doctor", "engineer", "farmer", "guard", "hunter", "judge", "lawyer"],
        "Beverage": ["coffee", "coke", "juice", "milk", "tea", "water", "wine", "beer", "soda", "lemonade"],
        "Color": ["red", "blue", "green", "yellow", "white", "black", "purple", "orange", "pink", "brown"],
        "Hobby": ["reading", "gaming", "hiking", "cooking", "fishing", "swimming", "drawing", "singing", "dancing", "running"]
    }
    
    kinds = sorted(kinds_dict)
    n_attributes = rows
    m_objects = columns - 1 
    
    # Randomly select attributes
    chosen_kinds = sorted(random.sample(kinds, k=n_attributes))
    table = [[kind] + random.sample(sorted(kinds_dict[kind]), k=m_objects) for kind in chosen_kinds]
    
    header = [str(i) for i in range(1, len(table[0]))]

    problem = {}
    puzzle_str = []
    
    for row in table:
        puzzle_str.append(f"{row[0]}: "+', '.join(sorted(row[1:])))
    
    try:
        premises = generate_puzzle_logic(table, level=level, minimal_conditions=True, max_seconds_for_minimizing=5)
    except Exception:
        premises = ["Clue generation failed. This is a fallback solvable puzzle."]
    
    indent = len(str(len(premises)))
    puzzle_str.append("\nClues:")
    for i, premise in enumerate(premises, 1):
        i = str(i).rjust(indent)
        puzzle_str.append(f"{i}. {premise}")
    
    puzzle_str.append("\nFill the following table to show your final answer.")
    puzzle_str.append(format_table_no_header(header, get_empty_table(table)))
    
    answer_str = format_table_no_header(header, table)
    
    problem['question'] = '\n'.join(puzzle_str)
    problem['answer'] = answer_str
    problem['attributes'] = [row[0] for row in table] 
    problem['raw_table'] = table # Important: Save raw data for contradiction injection
    return problem

# -----------------------------------------------------------------------------
# 3. Data Processing & Contradiction Injection (Revised)
# -----------------------------------------------------------------------------

def string_to_md5(s):
    encoded_string = s.encode('utf-8')
    md5_hash = hashlib.md5()
    md5_hash.update(encoded_string)
    return md5_hash.hexdigest()

def transform_problem_to_meta(problem, idx, language, split):
    timestamp = str(time.time())  
    id_string = f"zebra_logic_{idx}_{timestamp}"
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
        "difficulty_level":  problem["difficulty_level"],
        "language": language
    }

def _extract_attr_values(question_text: str) -> List[Tuple[str, List[str]]]:
    attrs: List[Tuple[str, List[str]]] = []
    for line in question_text.splitlines():
        m = re.match(r"^\s*([A-Za-z\-]+):\s*(.+)$", line)
        if not m:
            continue
        attr = m.group(1).strip()
        if attr in ["Clues", "Fill"]: 
            continue
        vals = [v.strip() for v in m.group(2).split(',') if v.strip()]
        if vals and len(vals) > 1: 
            attrs.append((attr, vals))
    return attrs

def inject_contradiction(problem_dict: dict) -> str:
    """
    Injects a contradiction by adding a clue that directly conflicts with the 
    ground truth spatial arrangement or attribute association.
    
    This avoids the "Option A vs Option B" style which models try to auto-correct.
    Instead, it asserts a "False Fact" as if it were a valid clue.
    """
    question_text = problem_dict['question']
    # Use raw table to understand ground truth
    # table structure: [[Attr1, Val1_1, Val1_2...], [Attr2, Val2_1, Val2_2...]]
    raw_table = problem_dict.get('raw_table')
    
    if not raw_table:
        # Fallback if raw_table is missing
        return question_text + "\nX. The puzzle is unsolvable."

    # Parse Clue Numbering to append correctly
    lines = question_text.splitlines()
    last_clue_num = 0
    insert_idx = len(lines)
    
    for i, line in enumerate(lines):
        m = re.match(r"^\s*(\d+)\.", line)
        if m:
            last_clue_num = int(m.group(1))
        if "Fill the following table" in line:
            insert_idx = i
            break
            
    next_clue_num = last_clue_num + 1
    
    # Strategy: "Spatial Inversion" or "False Neighbor"
    # Pick two distinct items (columns)
    # Item A is at Index i, Item B is at Index j (where i < j)
    # Truth: A is left of B.
    # False Clue: "A is to the right of B" or "A is immediately right of B"
    
    num_items = len(raw_table[0]) - 1 # excluding header
    if num_items < 2:
        return question_text # Cannot inject spatial contradiction
        
    idx_a = random.randint(0, num_items - 2)
    idx_b = random.randint(idx_a + 1, num_items - 1)
    
    # Pick two different attributes for the description to make it non-obvious
    # e.g. "The [Color of A] is to the right of the [Food of B]"
    attr_rows = list(range(len(raw_table)))
    row_idx_1 = random.choice(attr_rows)
    row_idx_2 = random.choice(attr_rows)
    
    # Values in raw_table are [Attr_Name, Val_Pos1, Val_Pos2...]
    # So Val at pos k is raw_table[row][k+1]
    
    val_a = raw_table[row_idx_1][idx_a + 1]
    attr_a_name = raw_table[row_idx_1][0]
    
    val_b = raw_table[row_idx_2][idx_b + 1]
    attr_b_name = raw_table[row_idx_2][0]
    
    # Truth: A (left) ... B (right)
    # Lie: A is to the right of B
    
    contradiction_type = random.choice(["right", "imm_right", "association"])
    
    if contradiction_type == "association":
        # Direct clash: "The [Val A] is [Val C]" (where Val C is actually at a different pos)
        # Find a value that exists but is NOT at idx_a
        wrong_idx = (idx_a + 1) % num_items
        wrong_val = raw_table[row_idx_2][wrong_idx + 1]
        fake_clue = f"{attr_a_name}:{val_a} == {attr_b_name}:{wrong_val}"
    
    elif contradiction_type == "right":
        fake_clue = f"{attr_a_name}:{val_a} is somewhere to the right of {attr_b_name}:{val_b}"
    
    else: # imm_right
        fake_clue = f"{attr_a_name}:{val_a} is immediately to the right of {attr_b_name}:{val_b}"

    # Format the new line
    indent = len(str(next_clue_num))
    new_line = f"{str(next_clue_num).rjust(indent)}. {fake_clue}"
    
    # Insert before the instruction line
    lines.insert(insert_idx, new_line)
    
    return "\n".join(lines)

def generate(count=100, difficulty='medium', language='en', split="train", **kwargs):
    force_solvable: Optional[bool] = kwargs.get("force_solvable")
    params = CONFIGS.get(difficulty, CONFIGS['medium'])
    
    rows = params["rows"]
    cols = params["columns"]
    level = params["level"]
    
    generated = 0
    idx = 0
    
    pbar = tqdm(total=count, desc=f"Gen {difficulty} (Solvable={force_solvable})")
    
    while generated < count:
        try:
            base = generator_zebra_logic_problem(rows=rows, columns=cols, level=level)
        except Exception:
            continue 
            
        base_dl = difficulty
        base['difficulty_level'] = base_dl

        if force_solvable is True:
            make_unsat = False
        elif force_solvable is False:
            make_unsat = True
        else:
            make_unsat = (random.random() < 0.5)

        if make_unsat:
            # Pass the full base dict to access raw_table
            q2 = inject_contradiction(base)
            answer_out = "unsolvable"
            out_problem = {"question": q2, "answer": answer_out, "difficulty_level": base_dl, "attributes": base.get('attributes', [])}
        else:
            out_problem = base
            answer_out = base['answer']

        attrs = out_problem.get('attributes', [])
        if not attrs:
            attrs = [x[0] for x in _extract_attr_values(out_problem['question'])]
        
        columns_desc = ", ".join([f"column {i+1} is {attr}" for i, attr in enumerate(attrs)])
        table_header = "| " + " | ".join(["Position"] + attrs) + " |"
        
        prompt_text = PROMPT_TEMPLATE.format(
            question=out_problem['question'],
            columns_desc=columns_desc,
            table_header=table_header
        )

        if make_unsat:
            prompt_text += "\nIf there is no valid solution, please output \\boxed{unsolvable} at the end.\n"

        meta = transform_problem_to_meta(out_problem, idx, language, split)
        
        item = {
            "prompt": prompt_text,
            "answer": answer_out,
            "task_name": DATASET_NAME,
            "ability": "logic_puzzle",
            "language": language,
            "meta": json.dumps(meta, ensure_ascii=False),
        }
        
        yield item
        generated += 1
        idx += 1
        pbar.update(1)
    
    pbar.close()

def save_to_jsonl(of_jsonl: str, of_meta: str, count: int, language: str, split: str, force_solvable: Optional[bool] = None):
    for path in (of_jsonl, of_meta):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    per = count // 3
    rem = count % 3
    diffs = ['easy', 'medium', 'hard']
    counts = {d: per for d in diffs}
    for d in diffs:
        if rem <= 0:
            break
        counts[d] += 1
        rem -= 1

    entries = []
    seen_keys = set()
    with open(of_jsonl, 'w', encoding='utf-8') as f1:
        for difficulty in diffs:
            num = counts[difficulty]
            if num <= 0:
                continue
            for item in generate(num, difficulty=difficulty, language=language, split=split, force_solvable=force_solvable):
                meta = json.loads(item['meta']) if isinstance(item.get('meta'), str) else item.get('meta')
                is_unsat = isinstance(item.get('answer'), str) and item.get('answer').strip().lower() == 'unsolvable'
                data_source = 'zebra_logic_unsolvable' if is_unsat else 'zebra_logic'
                prompt_content = item['prompt'] + "\nIf there is no solution, please output \\boxed{unsolvable} at the end.\n"
                
                key = json.dumps({"prompt": prompt_content, "answer": item.get('answer')}, ensure_ascii=False)
                if key in seen_keys:
                    continue
                entry = {
                    "data_source": data_source,
                    "prompt": [{"content": prompt_content, "role": "user"}],
                    "ability": item.get('ability', 'logic_puzzle'),
                    "reward_model": {"ground_truth": "unsolvable" if is_unsat else meta.get('answer'), "style": "unsolvable_generated" if is_unsat else "generated"},
                    "extra_info": {"index": meta.get('id', f"idx_{random.randint(0, int(1e9))}"), "question": meta.get('question')}
                }
                f1.write(json.dumps(entry, ensure_ascii=False) + '\n')
                entries.append(entry)
                seen_keys.add(key)

    try:
        parquet_path = of_jsonl.rsplit('.', 1)[0] + '.parquet'
        pd.DataFrame(entries).to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f"Warning: failed to write parquet for {of_jsonl}: {e}")

# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate Solvable Test Set
    print("Generating Solvable Test Data...")
    save_to_jsonl(
        'test_en_zebra_logic.jsonl', 
        'test_en_zebra_logic_raw.jsonl', 
        count=100, 
        language='en', 
        split='eval', 
        force_solvable=True
    )

    # Generate Unsolvable Test Set
    print("Generating Unsolvable Test Data...")
    save_to_jsonl(
        'test_en_zebra_logic_unsolvable.jsonl', 
        'test_en_zebra_logic_raw_unsolvable.jsonl', 
        count=100, 
        language='en', 
        split='eval', 
        force_solvable=False
    )

    print('Zebra-logic datasets generated successfully.')