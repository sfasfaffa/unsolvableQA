import time
import random
import json
from hashlib import sha256
from pysat.formula import CNF
from pysat.solvers import Solver
from tqdm import tqdm
import hashlib
import pandas as pd
# from .template import PROMPT_TEMPLATE
PROMPT_TEMPLATE = """
You are tasked with solving a Hamiltonian Cycle Puzzle.

### Rules:
A **Hamiltonian Cycle** in an undirected graph is a cycle that visits every vertex exactly once and returns to the starting vertex. The task is to determine whether a Hamiltonian Cycle exists in the given graph.

The graph is represented as follows:
- The first line contains a single integer `N`, which is the number of vertices in the graph.
- The subsequent lines each describe an edge in the graph. Each edge is represented by two space-separated integers `u` and `v`, which indicate that there is an undirected edge between vertex `u` and vertex `v`.
- The vertices are numbered from `0` to `N-1`.

### Response Format:
- Please output your answer within a code block (```) as follows:
```
<result>
```
- If a Hamiltonian Cycle exists, <result> should be a list of vertex indices that form the cycle, where the last vertex is the same as the first vertex to complete the cycle, for example:
```
[0, 2, 3, 1, 0]
```

Here is the puzzle:
{question} 
""".strip()

PROMPT_TEMPLATE_ZH = """
你的任务是解决一个哈密顿回路（Hamiltonian Cycle）问题。

### 规则：
1. 哈密顿回路是无向图中的一个回路，该回路恰好访问每个顶点一次，并返回到起始顶点。任务是判断给定的图中是否存在哈密顿回路。

2. 图的表示方式如下：
- 第一行包含一个整数 `N`，表示图中的顶点数量。
- 后续每一行描述图中的一条边。每条边由两个以空格分隔的整数 `u` 和 `v` 表示，表示在顶点 `u` 和顶点 `v` 之间存在一条无向边。
- 顶点编号从 `0` 到 `N-1`。

### 回答格式：
- 输出格式为 JSON 格式：
```json
{{
  "answer": "<result>"
}}
```
- 如果哈密顿回路存在，<result> 应该是一个数值列表，表示形成回路的顶点顺序，例如 [0, 2, 3, 1, 0]（其中最后一个顶点与第一个第一个顶点相同形成回路）。
- 如果不存在哈密顿回路，<result> 应该是 "NO"。

请解决以下的题目：
{question}
""".strip()

PUZZLE_TYPE = "graph_puzzle"
SOURCE_URL = "auto_generated"
DATASET_NAME = "hamiltonian_cycle"

class TimeoutException(Exception):
    pass

def generate_unique_seed():
    """Generate a unique random seed"""
    return int(sha256(str(random.random()).encode()).hexdigest(), 16)

def has_hamiltonian_cycle(num_nodes, edges, timeout=1):
    """Determine if a Hamiltonian Cycle exists"""
    start_time = time.time()

    graph = {i: set() for i in range(num_nodes)}
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)

    for node, neighbors in graph.items():
        if len(neighbors) < 2:
            return {"reason": f"Node {node} has degree {len(neighbors)} (less than 2)."}, False

    cnf = CNF()
    nodes = range(num_nodes)
    var = lambda i, j: i * num_nodes + j + 1  

    cnf.append([var(0, 0)])

    for i in nodes:
        if time.time() - start_time > timeout:
            raise TimeoutException("Hamiltonian cycle computation timed out.")
        cnf.append([var(i, j) for j in nodes])
        for j in range(num_nodes):
            for k in range(j + 1, num_nodes):
                cnf.append([-var(i, j), -var(i, k)])

    for j in nodes:
        if time.time() - start_time > timeout:
            raise TimeoutException("Hamiltonian cycle computation timed out.")
        cnf.append([var(i, j) for i in nodes])
        for i in range(num_nodes):
            for k in range(i + 1, num_nodes):
                cnf.append([-var(i, j), -var(k, j)])

    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    for j in range(num_nodes - 1):
        if time.time() - start_time > timeout:
            raise TimeoutException("Hamiltonian cycle computation timed out.")
        for i in nodes:
            for k in nodes:
                if (min(i, k), max(i, k)) not in edge_set:
                    cnf.append([-var(i, j), -var(k, j + 1)])

    for i in nodes:
        for k in nodes:
            if time.time() - start_time > timeout:
                raise TimeoutException("Hamiltonian cycle computation timed out.")
            if (min(i, k), max(i, k)) not in edge_set:
                cnf.append([-var(i, num_nodes - 1), -var(k, 0)])

    with Solver(name='glucose3') as solver:
        solver.append_formula(cnf)

        if time.time() - start_time > timeout:
            raise TimeoutException("Hamiltonian cycle computation timed out before solving.")
        
        if solver.solve():
            model = solver.get_model()
            cycle = [None] * num_nodes
            for j in nodes:
                for i in nodes:
                    if model[var(i, j) - 1] > 0:
                        cycle[j] = i
            return cycle, True
        else:
            return None, False

def generate_hamiltonian_cycle_problem(num_nodes_range=None, edge_density=None, ensure_hamiltonian=None):
    while True:
        problem = {}
        # If ensure_hamiltonian is provided, use it; otherwise randomize
        if ensure_hamiltonian is None:
            ensure_hamiltonian = random.choice([True, False])
        else:
            ensure_hamiltonian = bool(ensure_hamiltonian)
        seed = generate_unique_seed()
        random.seed(seed)

        if num_nodes_range is None:
            num_nodes_range = (8, 12)  

        num_nodes = random.randint(num_nodes_range[0], num_nodes_range[1])
        
        if edge_density is None:
            edge_density = random.uniform(0.6, 0.9)

        assert 0 <= edge_density <= 1, "Edge density must be between 0 and 1"
        assert num_nodes > 0, "Number of nodes must be greater than 0"

        edges = set()
        reasons = []

        max_possible_edges = num_nodes * (num_nodes - 1) // 2
        target_edges = min(int(max_possible_edges * edge_density), max_possible_edges)

        if ensure_hamiltonian:
            nodes = list(range(num_nodes))
            random.shuffle(nodes)
            for i in range(num_nodes):
                u, v = nodes[i], nodes[(i + 1) % num_nodes]
                edges.add((min(u, v), max(u, v)))

            while len(edges) < target_edges:
                u, v = sorted(random.sample(range(num_nodes), 2))
                if (u, v) not in edges:
                    edges.add((u, v))
        else:
            issues = [
                "cycle_with_missing_connection",
                "isolated_nodes",
                "dead_ends",
                "sparse_graph",
                "multiple_small_cycles",
                "unbalanced_degree_distribution",
                "critical_bridge_node",
            ]
            selected_issue = random.choice(issues)

            if selected_issue == "isolated_nodes":
                num_isolated_nodes = random.randint(1, max(1, num_nodes // 5))
                isolated_nodes = random.sample(range(num_nodes), num_isolated_nodes)
                reasons.append(f"isolated_nodes: {isolated_nodes}")
                non_isolated_nodes = [n for n in range(num_nodes) if n not in isolated_nodes]
                while len(edges) < target_edges:
                    u, v = sorted(random.sample(non_isolated_nodes, 2))
                    edges.add((u, v))

            elif selected_issue == "cycle_with_missing_connection":
                nodes = list(range(num_nodes))
                random.shuffle(nodes)
                for i in range(num_nodes):
                    u, v = nodes[i], nodes[(i + 1) % num_nodes]
                    edges.add((min(u, v), max(u, v)))
                num_removed_edges = random.randint(1, 3)
                for _ in range(num_removed_edges):
                    if edges:
                        edge_to_remove = random.choice(list(edges))
                        edges.remove(edge_to_remove)
                reasons.append(f"cycle_with_missing_connection: removed {num_removed_edges} edges")

            elif selected_issue == "multiple_small_cycles":
                num_cycles = random.randint(2, 4)
                subgraph_sizes = [num_nodes // num_cycles] * num_cycles
                subgraph_sizes[0] += num_nodes % num_cycles 
        
                start = 0
                for size in subgraph_sizes:
                    cycle_nodes = list(range(start, start + size))
                    if size < 2:
                        continue
                    for i in range(size):
                        u, v = cycle_nodes[i], cycle_nodes[(i + 1) % size]
                        edges.add((min(u, v), max(u, v)))
                    start += size

                reasons.append("multiple_small_cycles")

            elif selected_issue == "unbalanced_degree_distribution":
                while len(edges) < target_edges and len(edges) < max_possible_edges:
                    u, v = sorted(random.sample(range(num_nodes), 2))
                    edges.add((u, v))

                low_degree_nodes = random.sample(range(num_nodes), random.randint(1, max(1, num_nodes // 5)))
                for node in low_degree_nodes:
                    connected_edges = [e for e in edges if node in e]
                    if len(connected_edges) > 1:
                        for e in connected_edges[1:]:
                            edges.remove(e)
                reasons.append(f"unbalanced_degree_distribution: {low_degree_nodes}")

            elif selected_issue == "multiple_small_cycles":
                num_cycles = random.randint(2, 4)
                subgraph_sizes = [num_nodes // num_cycles] * num_cycles
                subgraph_sizes[0] += num_nodes % num_cycles 

                start = 0
                for size in subgraph_sizes:
                    cycle_nodes = list(range(start, start + size))
                    if size < 2:
                        continue
                    for i in range(size):
                        u, v = cycle_nodes[i], cycle_nodes[(i + 1) % size]
                        edges.add((min(u, v), max(u, v)))
                    start += size

                reasons.append("multiple_small_cycles")

            elif selected_issue == "dead_ends":
                while len(edges) < target_edges:
                    u, v = sorted(random.sample(range(num_nodes), 2))
                    edges.add((u, v))

                dead_ends = []
                for _ in range(random.randint(1, max(1, num_nodes // 10))):
                    node = random.choice(range(num_nodes))
                    connected_edges = [e for e in edges if node in e]
                    if len(connected_edges) > 1:
                        for e in connected_edges[1:]:
                            edges.remove(e)
                        dead_ends.append(node)

                reasons.append(f"dead_ends: {dead_ends}")

            elif selected_issue == "sparse_graph":
                while len(edges) < target_edges // 2: 
                    u, v = sorted(random.sample(range(num_nodes), 2))
                    edges.add((u, v))

                reasons.append(f"sparse_graph with edge_density={edge_density/2}")

            elif selected_issue == "critical_bridge_node":
                bridge_node = random.choice(range(num_nodes))
                left_subgraph = [n for n in range(num_nodes) if n != bridge_node][:num_nodes // 2]
                right_subgraph = [n for n in range(num_nodes) if n != bridge_node][num_nodes // 2:]

                for subgraph in [left_subgraph, right_subgraph]:
                    while len(edges) < target_edges // 2:
                        u, v = sorted(random.sample(subgraph, 2))
                        edges.add((u, v))

                for subgraph in [left_subgraph, right_subgraph]:
                    if len(edges) < target_edges:
                        u = bridge_node
                        v = random.choice(subgraph)
                        edges.add((min(u, v), max(u, v)))

                reasons.append(f"critical_bridge_node: {bridge_node}")

        edges = list(edges)[:max_possible_edges]
        output = [f"{num_nodes}"]
        output.extend([f"{u} {v}" for u, v in edges])
        question = "\n".join(output)

        try:
            cycle, verify = has_hamiltonian_cycle(num_nodes, edges, timeout=1)
        except TimeoutException:
            print("Timeout while verifying Hamiltonian Cycle. Retrying...")
            continue  
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue
        if isinstance(cycle, list):
            cycle.append(cycle[0])
        problem["question"] = question
        problem["answer"] = cycle if verify else "NO"
        problem["reason"] = cycle if verify else reasons
        return problem

def string_to_md5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def transform_problem_to_meta(problem, idx, language, split):
    timestamp = str(time.time())
    id_string = f"hamiltonian_path_{idx}_{timestamp}"
    hash_id_string = string_to_md5(id_string)
    return {
        "id": hash_id_string,
        "question": problem["question"],
        "answer": problem["answer"],
        "rationale": problem["reason"],
        "split": split,
        "type": PUZZLE_TYPE,
        "source_url": SOURCE_URL,
        "dataset_name": DATASET_NAME,
        "difficulty_level": problem.get("difficulty_level", "medium"),
        "language": language,
    }

difficulty_mappings = {
    "easy": {"num_nodes_range": (10, 15), "edge_density": 0.2},
    "medium": {"num_nodes_range": (15, 20), "edge_density": 0.3},
    "hard": {"num_nodes_range": (20, 25), "edge_density": 0.4},
}

def generate(count=10, difficulty='medium', language='en', split="train", **kwargs):
    prompt_template = PROMPT_TEMPLATE
    #split = kwargs.get("split", "eval")
    params = difficulty_mappings[difficulty]
    force_solvable = kwargs.get('force_solvable', None)
    for i in tqdm(range(count)):
        problem = generate_hamiltonian_cycle_problem(**params, ensure_hamiltonian=force_solvable)
        problem["difficulty_level"] = difficulty
        meta = transform_problem_to_meta(problem, i, language, split)
        # normalize answer token: map NO (legacy) to standardized 'unsolvable'
        answer_out = meta["answer"]
        if isinstance(answer_out, str) and answer_out.strip() == "NO":
            answer_out = "unsolvable"

        yield {
            "prompt": prompt_template.format(question=meta["question"]),
            "answer": answer_out,
            "task_name": DATASET_NAME,
            "ability": PUZZLE_TYPE,
            "language": language,
            "meta": json.dumps(meta),
        }

def save_to_jsonl(output_file, count, language, split, force_solvable=None):
    per = count // 3
    rem = count % 3
    counts = {'easy': per, 'medium': per, 'hard': per}
    if rem > 0:
        counts['easy'] += 1
        rem -= 1
    if rem > 0:
        counts['medium'] += 1

    entries = []
    with open(output_file, 'w', encoding='utf-8') as f:
        for difficulty in ["easy", "medium", "hard"]:
            num = counts[difficulty]
            if force_solvable is False:
                written_local = 0
                trials = 0
                max_trials = max(1000, num * 100)
                while written_local < num and trials < max_trials:
                    trials += 1
                    for item in generate(1, difficulty=difficulty, language=language, split=split, force_solvable=False):
                        meta = json.loads(item['meta']) if isinstance(item.get('meta'), str) else item.get('meta')
                        try:
                            lines = meta['question'].strip().split('\n')
                            n = int(lines[0])
                            edges = [tuple(map(int, ln.split())) for ln in lines[1:] if ln.strip()]
                            cycle, has = has_hamiltonian_cycle(n, edges, timeout=1)
                        except Exception:
                            has = True

                        if not has:
                            data_source = f"hamiltonian_cycle_unsolvable"
                            prompt_content = item['prompt'] + "\nIf you believe the graph has no Hamiltonian Cycle, please output \\boxed{unsolvable} at the end.\n"
                            entry = {
                                "data_source": data_source,
                                "prompt": [{"content": prompt_content, "role": "user"}],
                                "ability": item.get('ability', PUZZLE_TYPE),
                                "reward_model": {"ground_truth": "unsolvable", "style": "unsolvable_generated"},
                                "extra_info": {"index": meta.get('id', f"idx_{written_local}"), "question": meta.get('question')}
                            }
                            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                            entries.append(entry)
                            written_local += 1
                        if written_local >= num:
                            break

                if written_local < num:
                    print(f"Warning: only wrote {written_local}/{num} verified unsolvable {difficulty} items (trials={trials}).")
                continue

            for item in generate(count // 3, difficulty=difficulty, language=language, split=split, force_solvable=force_solvable):
                meta = json.loads(item['meta']) if isinstance(item.get('meta'), str) else item.get('meta')
                is_unsolvable = (isinstance(item.get('answer'), str) and item.get('answer').strip() == 'unsolvable')
                data_source = f"hamiltonian_cycle_unsolvable" if is_unsolvable else "hamiltonian_cycle"
                prompt_content = item['prompt'] + ("\nIf you believe the graph has no Hamiltonian Cycle, please output \\boxed{unsolvable} at the end.\n")
                entry = {
                    "data_source": data_source,
                    "prompt": [{"content": prompt_content, "role": "user"}],
                    "ability": item.get('ability', PUZZLE_TYPE),
                    "reward_model": {"ground_truth": "unsolvable" if is_unsolvable else meta.get('answer'), "style": "unsolvable_generated" if is_unsolvable else "generated"},
                    "extra_info": {"index": meta.get('id', f"idx_{random.randint(0, int(1e9))}"), "question": meta.get('question')}
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                entries.append(entry)
    try:
        pd.DataFrame(entries).to_parquet(output_file.rsplit('.',1)[0] + '.parquet', index=False)
    except Exception as e:
        print(f"Warning: failed to write parquet for {output_file}: {e}")

if __name__ == "__main__":
    # Normal files: prefer solvable examples
    save_to_jsonl('train_en_hamiltonian_cycle.jsonl', 50, language='en', split="train", force_solvable=True)
    # Unsolvable files: explicitly request unsolvable (force_solvable=False)
    save_to_jsonl('train_en_hamiltonian_cycle_unsolvable.jsonl', 50, language='en', split="train", force_solvable=False)
    save_to_jsonl('test_en_hamiltonian_cycle.jsonl', 50, language='en', split="eval", force_solvable=True)
    save_to_jsonl('test_en_hamiltonian_cycle_unsolvable.jsonl', 50, language='en', split="eval", force_solvable=False)
