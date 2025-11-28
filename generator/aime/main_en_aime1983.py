"""
Adapted pipeline for AIME data (test_aime1983.parquet).
Reads parquet file whose records follow the same logical structure as math500_test.jsonl
(i.e., fields like 'problem' and 'answer'). Samples N problems (default 200), generates
unsolvable variants with `UnsolvableProblemGenerator`, and writes outputs:
- unsolvable_aime1983_sample{N}_results.json  (full structured results)
- unsolvable_aime1983_sample{N}_results.jsonl (one JSON entry per line, filtered)
- unsolvable_aime1983_sample{N}_results.parquet

Usage:
    python main_en_aime1983.py [sample_size] [n_threads]

Example:
    python main_en_aime1983.py 200 8

Notes:
- Ensures extra_info['index'] is an integer (original dataframe index saved as index).
- Filters out entries whose prompt content starts with '[Reflection Advice]'.
"""

import os
import json
import random
import re
import numbers
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from distill_main_en import UnsolvableProblemGenerator

AIME_PARQUET = r"D:\dlearning\distill_0926\aime_data\base\test_aime1983.parquet"
OUT_JSON = "unsolvable_aime1983_sample{n}_results_1103.json"
OUT_JSONL = "unsolvable_aime1983_sample{n}_results_1103.jsonl"
OUT_PARQUET = "unsolvable_aime1983_sample{n}_results_1103.parquet"

# Mapping heuristics for columns if names differ
PREFERRED_PROBLEM_COLS = ["problem", "prompt", "question", "content"]
PREFERRED_ANSWER_COLS = ["answer", "solution", "answer_text", "ground_truth"]


def load_aime_parquet(path):
    print(f"Loading AIME parquet from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"AIME parquet not found at: {path}")
    df = pd.read_parquet(path)
    cols = {c.lower(): c for c in df.columns}

    # Heuristics to pick problem and answer columns by sampling values
    def score_problem_col(col):
        sample = df[col].dropna().head(100).astype(str)
        if sample.empty:
            return 0.0
        avg_len = sample.str.len().mean()
        has_latex = sample.str.contains(r"\\\\|\\\$|\$|\\\(|\\\)", regex=True).sum()
        # prefer long text and LaTeX-like content
        return avg_len * 0.6 + has_latex * 30

    def score_answer_col(col):
        sample = df[col].dropna().head(200)
        if sample.empty:
            return 0.0
        # if nested structure, try to inspect inner fields
        cnt_numeric = 0
        cnt_short = 0
        total = 0
        for v in sample:
            total += 1
            s = str(v).strip()
            if re.fullmatch(r"-?\d+(\.\d+)?", s):
                cnt_numeric += 1
            if len(s) <= 6:
                cnt_short += 1
        return (cnt_numeric / total) * 100 + (cnt_short / total) * 20
    # initial candidates from preferred names
    problem_col = None
    for name in PREFERRED_PROBLEM_COLS:
        if name in cols:
            problem_col = cols[name]
            break
    answer_col = None
    for name in PREFERRED_ANSWER_COLS:
        if name in cols:
            answer_col = cols[name]
            break
    # If not found, score all object-like columns
    if problem_col is None:
        best = (None, -1)
        for c in df.columns:
            try:
                sc = score_problem_col(c)
            except Exception:
                sc = 0
            if sc > best[1]:
                best = (c, sc)
        problem_col = best[0]
    if answer_col is None:
        best = (None, -1)
        for c in df.columns:
            if c == problem_col:
                continue
            try:
                sc = score_answer_col(c)
            except Exception:
                sc = 0
            if sc > best[1]:
                best = (c, sc)
        answer_col = best[0]
    if problem_col is None:
        raise ValueError(f"Could not find a problem column in {path}. Available columns: {list(df.columns)}")
    # Build list of problem dicts, handling nested structures
    problems = []
    # helper: if a cell is a JSON string, parse it to native structure
    def try_parse_json_cell(v):
        try:
            if isinstance(v, str):
                s = v.strip()
                if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
                    try:
                        return json.loads(s)
                    except Exception:
                        return v
            return v
        except Exception:
            return v
    for idx, row in df.iterrows():
        # extract problem text
        prob_raw = row.get(problem_col, "") if problem_col in df.columns else ""
        prob_raw = try_parse_json_cell(prob_raw)
        prob_text = ""
        try:
            if isinstance(prob_raw, (list, dict)):
                # try to extract prompt[0].content or 'question' key
                if isinstance(prob_raw, list) and len(prob_raw) > 0 and isinstance(prob_raw[0], dict):
                    prob_text = prob_raw[0].get('content', '') or prob_raw[0].get('question', '')
                elif isinstance(prob_raw, dict):
                    prob_text = prob_raw.get('prompt', '') or prob_raw.get('question', '') or prob_raw.get('content', '')
                    # if prompt is a list inside dict
                    if isinstance(prob_text, list) and len(prob_text) > 0 and isinstance(prob_text[0], dict):
                        prob_text = prob_text[0].get('content', '')
            else:
                prob_text = str(prob_raw)
        except Exception:
            prob_text = str(prob_raw)

        # Strict extraction: require `reward_model.ground_truth` to be present somewhere in the row
        ground = None
        for c in df.columns:
            try:
                v = try_parse_json_cell(row.get(c))
                if isinstance(v, dict):
                    # prefer reward_model.ground_truth when present
                    rm = v.get('reward_model') if isinstance(v.get('reward_model'), dict) else None
                    if rm is not None and 'ground_truth' in rm:
                        ground = rm.get('ground_truth')
                        break
                    # also accept top-level 'ground_truth' key in dicts (less preferred)
                    if 'ground_truth' in v:
                        ground = v.get('ground_truth')
                        break
            except Exception:
                continue

        if ground is None:
            raise RuntimeError(f"Failed to locate 'reward_model.ground_truth' for row idx={idx}; aborting to avoid bad labels.")

        # enforce integer-only ground truth (digits only)
        gstr = str(ground).strip()
        # If ground contains non-digit text like '080 or 081 (both were accepted)',
        # extract the first numeric token and convert to int (strip leading zeros).
        m = re.search(r"(\d+)", gstr)
        if not m:
            raise RuntimeError(f"Invalid ground_truth format for idx={idx}: {gstr!r}. Expected integer digits somewhere in the string.")
        num = int(m.group(1))
        ans = str(num)
        if ans != gstr:
            print(f"[WARN] Normalized nonstandard ground_truth for idx={idx}: {gstr!r} -> {ans}")

        # ensure unique_id is int when possible, otherwise keep original index
        # normalize index: prefer integer when possible
        if isinstance(idx, numbers.Integral):
            uid = int(idx)
        else:
            try:
                uid = int(str(idx))
            except Exception:
                uid = idx

        problems.append({
            "problem": str(prob_text).strip(),
            "answer": str(ans).strip(),
            "unique_id": uid
        })
    print(f"Detected problem_col={problem_col}, answer_col={answer_col}")
    return problems


def is_valid_problem_entry(entry):
    if not entry or "prompt" not in entry or not entry["prompt"]:
        return False
    content = entry["prompt"][0].get("content", "").strip()
    if not content or content.startswith("[Reflection Advice]"):
        return False
    return True


def process_problem(idx, item, contradiction_type="Axiom Contradiction", difficulty=5):
    print(f"\n===== Processing AIME problem {idx+1} (uid={item['unique_id']}) =====\n")
    gen = UnsolvableProblemGenerator(model_name="deepseek-chat")
    item["problem"] = item["problem"].replace("[{'content': '", "").replace("'}]", "").replace("', 'role': 'user", "")
    item['problem'] = item['problem'].replace("[{'content': \"", "").replace("'}]", "").replace(", 'role': 'user", "")
    config = {
        "solvable_problem": item["problem"],
        "ground_truth_answer": item["answer"],
        "problem_type": "math_problem",
        "contradiction_type": contradiction_type,
        "difficulty": difficulty
    }
    result = gen.run_pipeline(config, max_retries=3)
    jsonl_entries = []
    # solvable
    jsonl_entries.append({
        "data_source": "aime",
        "prompt": [{"content": item["problem"] + "\nIf you believe the maze is unsolvable, please output \\boxed{unsolvable} at the end.\n", "role": "user"}],
        "ability": "MATH",
        "reward_model": {"ground_truth": item["answer"], "style": "aime_format"},
        "extra_info": {"index": int(item.get("unique_id", idx))}
    })
    # unsolvable variant
    if result and result.get("status") == "success" and result.get("unsolvable_problem"):
        unsolvable_problem = result["unsolvable_problem"]
        jsonl_entries.append({
            "data_source": "aime_unsolvable",
            "prompt": [{"content": unsolvable_problem + "\nIf you believe the maze is unsolvable, please output \\boxed{unsolvable} at the end.\n", "role": "user"}],
            "ability": "MATH",
            "reward_model": {"ground_truth": "unsolvable", "style": "unsolvable_generated"},
            "extra_info": {"index": int(item.get("unique_id", idx))}
        })
    return {"unique_id": item.get("unique_id", idx), "jsonl_entries": jsonl_entries, "result": result}


def main(sample_size=200, n_threads=8, contradiction_type="Axiom Contradiction", difficulty=5):
    print(f"Starting AIME pipeline: sample_size={sample_size}, n_threads={n_threads}, parquet={AIME_PARQUET}")
    try:
        problems = load_aime_parquet(AIME_PARQUET)
    except Exception as e:
        print(f"Error loading AIME parquet: {e}")
        raise
    if sample_size is None or sample_size > len(problems):
        sample_size = len(problems)
    print(f"Total problems available: {len(problems)}; sampling: {sample_size}")
    sampled = random.sample(problems, sample_size)
    results = []
    jsonl_all = []
    
    # Try to use tqdm for progress bar, fall back to manual progress if unavailable
    try:
        from tqdm import tqdm as progress_bar
        use_tqdm = True
    except ImportError:
        progress_bar = None
        use_tqdm = False
        print("tqdm not available, using manual progress tracking")
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(process_problem, i, item, contradiction_type, difficulty): i for i, item in enumerate(sampled)}
        
        if use_tqdm and progress_bar:
            # Use tqdm progress bar
            for future in progress_bar(as_completed(futures), total=len(futures), desc="Processing"):
                i = futures[future]
                try:
                    res = future.result()
                except Exception as e:
                    print(f"Error processing item {i}: {e}")
                    continue
                results.append(res)
                if res.get("jsonl_entries"):
                    # always append solvable entry
                    jsonl_all.append(res["jsonl_entries"][0])
                    if len(res["jsonl_entries"]) > 1 and res.get("result", {}).get("status") == "success":
                        jsonl_all.append(res["jsonl_entries"][1])
        else:
            # Manual progress tracking
            completed = 0
            for future in as_completed(futures):
                i = futures[future]
                try:
                    res = future.result()
                except Exception as e:
                    print(f"Error processing item {i}: {e}")
                    continue
                results.append(res)
                if res.get("jsonl_entries"):
                    # always append solvable entry
                    jsonl_all.append(res["jsonl_entries"][0])
                    if len(res["jsonl_entries"]) > 1 and res.get("result", {}).get("status") == "success":
                        jsonl_all.append(res["jsonl_entries"][1])
                completed += 1
                print(f"Progress: {completed}/{len(futures)} ({100*completed//len(futures)}%)")
    # sort results
    results.sort(key=lambda x: x["unique_id"]) 
    # write full results
    out_json = OUT_JSON.format(n=sample_size)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # write filtered jsonl
    out_jsonl = OUT_JSONL.format(n=sample_size)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for entry in jsonl_all:
            if is_valid_problem_entry(entry):
                # ensure int index
                if "extra_info" in entry and isinstance(entry["extra_info"], dict):
                    try:
                        entry["extra_info"]["index"] = int(entry["extra_info"].get("index", 0))
                    except Exception:
                        entry["extra_info"]["index"] = 0
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # convert jsonl to parquet via pandas.json_normalize
    df = pd.json_normalize([e for e in jsonl_all if is_valid_problem_entry(e)])
    out_parquet = OUT_PARQUET.format(n=sample_size)
    df.to_parquet(out_parquet, index=False)
    print(f"Wrote {len(jsonl_all)} entries (filtered to {len(df)}) to:\n  {out_json}\n  {out_jsonl}\n  {out_parquet}")


if __name__ == '__main__':
    import sys
    sample_size = 50
    n_threads = 50
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
        except Exception:
            pass
    if len(sys.argv) > 2:
        try:
            n_threads = int(sys.argv[2])
        except Exception:
            pass
    try:
        main(sample_size=sample_size, n_threads=n_threads)
    except Exception as e:
        import traceback
        print("Script failed with exception:")
        traceback.print_exc()
