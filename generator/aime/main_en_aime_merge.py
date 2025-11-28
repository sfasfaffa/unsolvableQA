"""
Merge two AIME parquet files, deduplicate by prompt content, and run the unsolvable generation
pipeline (same behavior as `main_en_aime1983.py`).

Usage:
    python main_en_aime_merge.py [sample_size] [n_threads]

Example:
    python main_en_aime_merge.py 200 8

Outputs:
  - unsolvable_aime25x24_sample{n}_results.json
  - unsolvable_aime25x24_sample{n}_results.jsonl
  - unsolvable_aime25x24_sample{n}_results.parquet
"""

import os
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# reuse helpers from the existing AIME pipeline
from main_en_aime1983 import is_valid_problem_entry, process_problem
import pandas as pd


# Local strict loader: read parquet and extract problem text and reward_model.ground_truth (must exist and be integer)
def load_parquet_strict(path):
    print(f"Strict-loading parquet: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)

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

    problems = []
    for idx, row in df.iterrows():
        # find problem text: prefer a dict with 'prompt' containing content
        prob_text = None
        for c in df.columns:
            try:
                v = try_parse_json_cell(row.get(c))
                if isinstance(v, dict):
                    # prompt may be a list of dicts with 'content'
                    p = v.get('prompt')
                    if isinstance(p, list) and len(p) > 0 and isinstance(p[0], dict):
                        ct = p[0].get('content')
                        if ct:
                            prob_text = str(ct)
                            break
                    if isinstance(p, str) and p.strip():
                        prob_text = str(p)
                        break
                    # fallback to known keys
                    for key in ('question', 'content', 'problem'):
                        if key in v and v.get(key):
                            prob_text = str(v.get(key))
                            break
                    if prob_text:
                        break
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    ct = v[0].get('content') or v[0].get('question')
                    if ct:
                        prob_text = str(ct)
                        break
            except Exception:
                continue
        if prob_text is None:
            # fallback: try common column names
            for name in ('prompt', 'problem', 'question', 'content'):
                if name in df.columns:
                    val = try_parse_json_cell(row.get(name))
                    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                        prob_text = val[0].get('content') or val[0].get('question')
                        if prob_text:
                            prob_text = str(prob_text)
                            break
                    elif val:
                        prob_text = str(val)
                        break
        if not prob_text:
            raise RuntimeError(f"Failed to extract problem text for row idx={idx} in {path}")

        # Strict: locate reward_model.ground_truth in any column dict
        ground = None
        for c in df.columns:
            try:
                v = try_parse_json_cell(row.get(c))
                if isinstance(v, dict):
                    # Case A: the column itself is the reward_model dict (contains ground_truth)
                    if 'ground_truth' in v:
                        ground = v.get('ground_truth')
                        break
                    # Case B: the column is a larger dict that contains a nested 'reward_model' dict
                    rm = v.get('reward_model')
                    if isinstance(rm, dict) and 'ground_truth' in rm:
                        ground = rm.get('ground_truth')
                        break
            except Exception:
                continue
        if ground is None:
            # Debug: print the raw row and per-column parsed values to help diagnose missing reward_model.ground_truth
            try:
                print(f"[DEBUG] Could not find reward_model.ground_truth for row idx={idx} in {path}")
                # Print raw row repr
                try:
                    raw_row = dict(row)
                    print("[DEBUG] raw row:", raw_row)
                except Exception:
                    raw_row = None
                    print("[DEBUG] raw row (repr):", repr(row))

                # Build parsed view per column and print
                parsed_columns = {}
                for c in df.columns:
                    try:
                        parsed = try_parse_json_cell(row.get(c))
                        parsed_columns[c] = parsed
                        print(f"[DEBUG] column '{c}': {type(parsed).__name__} -> {parsed}")
                    except Exception as e:
                        parsed_columns[c] = f"<error: {e}>"
                        print(f"[DEBUG] column '{c}': error parsing: {e}")

                # Also write a JSON debug file with the parsed_columns and a safe raw row representation
                try:
                    debug_obj = {
                        'path': path,
                        'row_idx': idx,
                        'raw_row': raw_row if raw_row is not None else repr(row),
                        'parsed_columns': parsed_columns
                    }
                    debug_filename = f"aime_merge_debug_row_{idx}.json"
                    with open(debug_filename, 'w', encoding='utf-8') as dfh:
                        json.dump(debug_obj, dfh, ensure_ascii=False, indent=2, default=str)
                    print(f"[DEBUG] Wrote parsed row debug to: {os.path.abspath(debug_filename)}")
                except Exception as e:
                    print(f"[DEBUG] Failed to write debug file: {e}")
            except Exception:
                pass
            raise RuntimeError(f"Failed to locate reward_model.ground_truth for row idx={idx} in {path}")

        gstr = str(ground).strip()
        m = re.search(r"(\d+)", gstr)
        if not m:
            raise RuntimeError(f"Invalid ground_truth format for idx={idx} in {path}: {gstr!r}")
        num = int(m.group(1))
        ans = str(num)
        if ans != gstr:
            print(f"[WARN] Normalized nonstandard ground_truth for idx={idx} in {path}: {gstr!r} -> {ans}")

        problems.append({
            'problem': prob_text.strip(),
            'answer': ans,
            'unique_id': idx
        })
    return problems


AIME_PARQUET_1 = r"D:\dlearning\distill_0926\aime_data\base\aime25x10.parquet"
AIME_PARQUET_2 = r"D:\dlearning\distill_0926\aime_data\base\aime24x10.parquet"
OUT_JSON = "unsolvable_aime25x24_sample{n}_results_1104_v3.json"
OUT_JSONL = "unsolvable_aime25x24_sample{n}_results_1104_v3.jsonl"
OUT_PARQUET = "unsolvable_aime25x24_sample{n}_results_1104_v3.parquet"


def merge_and_dedupe(parquet1, parquet2):
    print(f"Loading parquet1: {parquet1}")
    p1 = load_parquet_strict(parquet1)
    print(f"Loading parquet2: {parquet2}")
    p2 = load_parquet_strict(parquet2)

    combined = p1 + p2
    # Validate/normalize answers: ensure integer ground truth for every loaded item
    for item in combined:
        ans = str(item.get("answer", "")).strip()
        if not re.fullmatch(r"\d+", ans):
            m = re.search(r"(\d+)", ans)
            if not m:
                raise RuntimeError(f"Failed to extract integer ground_truth for merged item unique_id={item.get('unique_id')} answer={ans!r}")
            num = int(m.group(1))
            new_ans = str(num)
            if new_ans != ans:
                print(f"[WARN] Normalized merged ground_truth for unique_id={item.get('unique_id')}: {ans!r} -> {new_ans}")
            item["answer"] = new_ans
    print(f"Combined records before dedupe: {len(combined)}")

    seen = set()
    deduped = []
    for item in combined:
        prob = (item.get("problem") or "").strip()
        key = re_normalize(prob)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    print(f"Records after dedupe: {len(deduped)} (removed {len(combined)-len(deduped)} duplicates)")

    # reassign unique_id sequentially to avoid conflicts
    for i, it in enumerate(deduped):
        it["unique_id"] = i
    return deduped


def re_normalize(s: str) -> str:
    # simple normalization for dedupe: strip, lower, collapse whitespace
    return " ".join(s.strip().lower().split())


def main(sample_size=200, n_threads=8, contradiction_type="Axiom Contradiction", difficulty=5):
    print(f"Merging AIME parquets and running pipeline: sample_size={sample_size}, n_threads={n_threads}")
    if not os.path.exists(AIME_PARQUET_1) or not os.path.exists(AIME_PARQUET_2):
        raise FileNotFoundError("One or both AIME parquet files not found. Please check paths.")

    deduped = merge_and_dedupe(AIME_PARQUET_1, AIME_PARQUET_2)

    if sample_size is None or sample_size > len(deduped):
        sample_size = len(deduped)
    print(f"Total unique problems available: {len(deduped)}; sampling: {sample_size}")

    sampled = random.sample(deduped, sample_size)

    results = []
    jsonl_all = []
    tqdm = None
    try:
        from tqdm import tqdm as _tqdm
        tqdm = _tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    total = len(sampled)
    pbar = None
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(process_problem, i, item, contradiction_type, difficulty): i for i, item in enumerate(sampled)}
        if use_tqdm and tqdm is not None:
            pbar = tqdm(total=total, desc="Processing", ncols=80)
        completed = 0
        for future in as_completed(futures):
            i = futures[future]
            try:
                res = future.result()
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                if pbar is not None:
                    pbar.update(1)
                else:
                    completed += 1
                    print(f"Progress: {completed}/{total}", end='\r', flush=True)
                continue
            results.append(res)
            if res.get("jsonl_entries"):
                jsonl_all.append(res["jsonl_entries"][0])
                if len(res["jsonl_entries"]) > 1 and res.get("result", {}).get("status") == "success":
                    jsonl_all.append(res["jsonl_entries"][1])
            if pbar is not None:
                pbar.update(1)
            else:
                completed += 1
                print(f"Progress: {completed}/{total}", end='\r', flush=True)
        if pbar is not None:
            pbar.close()

    # sort results
    results.sort(key=lambda x: x["unique_id"]) 

    out_json = OUT_JSON.format(n=sample_size)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    out_jsonl = OUT_JSONL.format(n=sample_size)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for entry in jsonl_all:
            if is_valid_problem_entry(entry):
                if "extra_info" in entry and isinstance(entry["extra_info"], dict):
                    try:
                        entry["extra_info"]["index"] = int(entry["extra_info"].get("index", 0))
                    except Exception:
                        entry["extra_info"]["index"] = 0
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # convert jsonl to parquet via pandas.json_normalize
    import pandas as pd
    df = pd.json_normalize([e for e in jsonl_all if is_valid_problem_entry(e)])
    out_parquet = OUT_PARQUET.format(n=sample_size)
    df.to_parquet(out_parquet, index=False)

    print(f"Wrote {len(jsonl_all)} entries (filtered to {len(df)}) to:\n  {out_json}\n  {out_jsonl}\n  {out_parquet}")


if __name__ == '__main__':
    import sys
    sample_size = 60
    n_threads = 60
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
