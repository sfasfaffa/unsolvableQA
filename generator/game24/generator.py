import random
import os
# from .template import PROMPT_TEMPLATE
from itertools import permutations, product, combinations
import pandas as pd
from fractions import Fraction
import json
from tqdm import tqdm
PROMPT_TEMPLATE = """Solution for the 24 Game

# Description:
You are given four or five or six integers, ranging from 1 to 13, provide an arithmetic expression that results in 24.

# Rules:

You must use all the given numbers, each exactly once.
The operators you can use include: addition (+), subtraction (-), multiplication (*), and division (/).
You can use parentheses to change the order of operations.

**Response format:**
- Please output your answer within a code block (```) as follows:
```
<result>
```
- If there is a solution, <result> is the sequence of numbers and operators that results in 24, for example: 
```
(8 / 2) * (8 - 2)
```

# Input:
{question}

Please provide a solution for the 24 game according to the above rules and input."""


def generate_numbers(num_nums):
    return [random.randint(1, 13) for _ in range(num_nums)]
operations = ['+', '-', '*', '/']

# Define error tolerance

# Use exact rational arithmetic to avoid floating point errors
def apply_op(a: Fraction, b: Fraction, op: str):
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '*':
        return a * b
    if op == '/':
        if b == 0:
            return None
        return a / b
    return None


def all_binary_expressions(values):
    """Generate all possible results and corresponding expression strings by building binary expression trees.
    values: list of (Fraction, str) pairs
    Returns a generator of (Fraction, str) results.
    """
    if len(values) == 1:
        yield values[0]
        return

    # partition into two non-empty groups
    n = len(values)
    # choose split positions using combinations of indices
    for k in range(1, n//1):
        # choose k elements for left
        for left_indices in combinations(range(n), k):
            left = [values[i] for i in left_indices]
            right = [values[i] for i in range(n) if i not in left_indices]
            # recursively compute
            for lv, ls in all_binary_expressions(left):
                for rv, rs in all_binary_expressions(right):
                    for op in operations:
                        res = apply_op(lv, rv, op)
                        if res is None:
                            continue
                        yield (res, f"({ls}{op}{rs})")


def can_form_24(nums, lang='en'):
    """Deterministically check whether nums can form 24 using exact rational arithmetic.
    Returns an answer string with the expression or 'cannot form 24'.
    """
    answer_cue = 'The answer is: ' if lang=='en' else 'The answer is: '
    refuse_cue = 'cannot form 24' if lang=='en' else 'cannot form 24'

    len_nums = len(nums)
    fracs = [Fraction(n) for n in nums]

    # try all permutations
    for perm in permutations(range(len(fracs))):
        vals = [(fracs[i], str(nums[i])) for i in perm]
        for res, expr in all_binary_expressions(vals):
            if res == 24:
                return f"{answer_cue}{expr} = 24"
    return refuse_cue

def generate(count=100, difficulty='medium', language='en', split="train", force_solvable=None):
    #param1 = kwargs.get('param1', default_value1)
    #param2 = kwargs.get('param2', default_value2)
    # Keep three difficulty labels but make 'hard' slightly easier (5 numbers) to speed generation.
    dic = {'easy': 4, 'medium': 5, 'hard': 5}
    num_nums = dic[difficulty]
    prompt_template = PROMPT_TEMPLATE
    #exist = {}
    for i in tqdm(range(count)):
        # Sample numbers and optionally enforce solvability/unsolvability
        attempts = 0
        max_attempts = 3000
        numbers = generate_numbers(num_nums)
        answer = can_form_24(numbers, language)

        def _is_solvable(ans: str) -> bool:
            try:
                return isinstance(ans, str) and ('cannot form 24' not in ans)
            except Exception:
                return False

        if force_solvable is True:
            while not _is_solvable(answer) and attempts < max_attempts:
                numbers = generate_numbers(num_nums)
                answer = can_form_24(numbers, language)
                attempts += 1
        elif force_solvable is False:
            while _is_solvable(answer) and attempts < max_attempts:
                numbers = generate_numbers(num_nums)
                answer = can_form_24(numbers, language)
                attempts += 1

        sorted_numbers = tuple(numbers) #sorted(numbers)
        numbers_str = ",".join(map(str, numbers))
        # normalize answer: if unsolvable according to can_form_24, return standardized token
        answer_out = answer
        if not _is_solvable(answer):
            answer_out = "unsolvable"

        #print(numbers, answer_out)
        yield {
            "prompt":prompt_template.format(question=numbers_str), 
            "answer":  answer_out,
            "task_name": "game24",    
            "ability": "logic_puzzle", 
            "language": language,
            "meta": json.dumps({
                "id":"game24_"+difficulty+"_"+str(i), #hash?
                "question": numbers,
                "answer": answer_out,
                "rationale": "", 
                "split": split,
                "type": "code_puzzle", 
                "source_url": "auto-generated", 
                "dataset_name": "game24", 
                "difficulty_level": difficulty,
                "language": language,
            }),            
        }

def save_to_jsonl(of1, of2, count, lange='en', force_solvable=None):
    """Save problems and metas to two JSONL files. Ensure parent dirs exist and distribute counts across difficulties."""
    for path in (of1, of2):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    per = count // 3
    rem = count % 3
    counts = {'easy': per, 'medium': per, 'hard': per}
    if rem > 0:
        counts['easy'] += 1
        rem -= 1
    if rem > 0:
        counts['medium'] += 1

    entries = []
    with open(of1, 'w', encoding='utf-8') as f1:
        for difficulty in ['easy', 'medium', 'hard']:
            num = counts[difficulty]
            # If we're producing an unsolvable file (force_solvable is False), only generate from 'easy'
            if force_solvable is False:
                # produce 'num' verified easy unsolvable problems
                written = 0
                trials = 0
                max_trials = max(1000, num * 50)
                while written < num and trials < max_trials:
                    # generate one candidate easy problem that prefers unsolvable
                    for item in generate(1, 'easy', lange, force_solvable=False):
                        trials += 1
                        meta = json.loads(item['meta']) if isinstance(item.get('meta'), str) else item.get('meta')
                        nums = meta.get('question')
                        # verify truly unsolvable
                        try:
                            verify_ans = can_form_24(nums, lange)
                            is_solvable = isinstance(verify_ans, str) and ('cannot form 24' not in verify_ans)
                        except Exception:
                            # If verification fails, conservatively treat as solvable and skip
                            is_solvable = True

                        if not is_solvable:
                            # label difficulty as 'easy' for unsolvable outputs
                            meta['difficulty_level'] = 'easy'
                            item['meta'] = json.dumps(meta, ensure_ascii=False)
                            # build output entry in requested schema
                            data_source = f"game24_unsolvable"
                            prompt_content = item['prompt'] + "\nIf there is no solution, please output \\boxed{unsolvable} at the end.\n"
                            entry = {
                                "data_source": data_source,
                                "prompt": [{"content": prompt_content, "role": "user"}],
                                "ability": item.get('ability', 'logic_puzzle'),
                                "reward_model": {"ground_truth": "unsolvable", "style": "unsolvable_generated"},
                                "extra_info": {"index": meta.get('id', f"idx_{written}"), "question": meta.get('question')}
                            }
                            f1.write(json.dumps(entry, ensure_ascii=False) + '\n')
                            entries.append(entry)
                            written += 1
                        if written >= num:
                            break

                if written < num:
                    print(f"Warning: only wrote {written}/{num} verified unsolvable {difficulty} items (trials={trials}).")
                continue

            gen_difficulty = 'medium' if difficulty == 'hard' else difficulty
            for item in generate(num, gen_difficulty, lange, force_solvable=force_solvable):
                # keep metadata label as the intended difficulty
                meta = json.loads(item['meta']) if isinstance(item.get('meta'), str) else item.get('meta')
                meta['difficulty_level'] = difficulty
                item['meta'] = json.dumps(meta, ensure_ascii=False)
                # Build standardized entry
                is_unsolvable = (isinstance(item.get('answer'), str) and item.get('answer').strip() == "unsolvable")
                data_source = f"game24_unsolvable" if is_unsolvable else "game24"
                prompt_content = item['prompt'] + ("\nIf there is no solution, please output \\boxed{unsolvable} at the end.\n")
                entry = {
                    "data_source": data_source,
                    "prompt": [{"content": prompt_content, "role": "user"}],
                    "ability": item.get('ability', 'logic_puzzle'),
                    "reward_model": {"ground_truth": "unsolvable" if is_unsolvable else meta.get('answer'), "style": "unsolvable_generated" if is_unsolvable else "generated"},
                    "extra_info": {"index": meta.get('id', f"idx_{random.randint(0, int(1e9))}"), "question": meta.get('question')}
                }
                f1.write(json.dumps(entry, ensure_ascii=False) + '\n')
                entries.append(entry)
    # write parquet file alongside the jsonl
    try:
        parquet_path = of1.rsplit('.', 1)[0] + '.parquet'
        df = pd.DataFrame(entries)
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f"Warning: failed to write parquet for {of1}: {e}")
if __name__ == "__main__":
    save_to_jsonl('test_en_game24.jsonl', 'test_en_game24_raw.jsonl', 50, 'en', force_solvable=True)
    save_to_jsonl('test_en_game24_unsolvable.jsonl', 'test_en_game24_raw_unsolvable.jsonl', 50, 'en', force_solvable=False)