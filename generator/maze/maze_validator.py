import os
import json
import openai
import re

def read_maze_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    maze_text = data.get('solvable_maze', '')
    maze = [[int(cell) for cell in line] for line in maze_text.strip().split('\n') if line]
    return maze

def ask_llm_for_solution(maze_text):
    os.environ["OPENAI_API_KEY"] = "sk-4093ceb1897e49e0bb1fbc6a8d754dab"
    os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"]
    )
    prompt = (
        "请为以下迷宫寻找一条从起点(S)到终点(E)的路径。"
        "请按照如下格式回答：[(x1,y1), (x2,y2), ..., (xn,yn)]，每一步是坐标，且每一步必须紧邻且不能走到墙里。"
        "注意：x代表列，y代表行，即(x,y)=(列,行)。请严格按照此顺序填写坐标。"
        f"迷宫如下：\n{maze_text}\nS为起点(1,1)，E为终点(宽度-2,高度-2)。"
    )
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=16384
    )
    answer = response.choices[0].message.content
    if not answer:
        return [], "LLM未返回答案"
    match = re.search(r'\[(.*?)\]', str(answer), re.DOTALL)
    if match:
        steps_str = match.group(1)
        steps = re.findall(r'\((\d+),(\d+)\)', steps_str)
        path = [(int(x), int(y)) for x, y in steps]  # x=列, y=行
        return path, answer
    return [], answer

def validate_path(maze, path):
    if not path:
        return False, "路径为空"
    height = len(maze)
    width = len(maze[0])
    for i, (x, y) in enumerate(path):
        if not (0 <= x < width and 0 <= y < height):
            return False, f"坐标({x},{y})越界"
        if maze[y][x] != 0:
            return False, f"坐标({x},{y})不是空地"
        if i > 0:
            px, py = path[i-1]
            if abs(px-x) + abs(py-y) != 1:
                return False, f"第{i}步({x},{y})与前一步({px},{py})不相邻"
    if path[0] != (1, 1):
        return False, "起点不是(1,1)"
    if path[-1] != (width-2, height-2):
        return False, f"终点不是({width-2},{height-2})"
    return True, "路径验证通过"

if __name__ == "__main__":
    maze = read_maze_from_json("solvable_maze.json")
    maze_text = '\n'.join(''.join(str(cell) for cell in row) for row in maze)
    path, llm_answer = ask_llm_for_solution(maze_text)
    print("LLM回答:", llm_answer)
    valid, msg = validate_path(maze, path)
    print("验证结果:", msg)