import json, re, ast
from llm_client import HelloAgentsLLM
from tools import ToolExecutor, search, calculator

# 新的提示模板，要求 LLM 只输出 JSON 对象，而不是普通文本。这种格式更适合程序化处理 LLM 的输出，尤其是在需要解析和调用工具的场景中。
REACT_PROMPT_TEMPLATE_JSON = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

你必须严格只输出一个 JSON 对象，不要输出任何其他文字，不要使用 Markdown 代码块。
- 需要调用工具时输出:
{{"thought": "你的思考过程", "action": {{"name": "工具名", "input": "工具输入"}}}}
- 已经得到最终答案时输出:
{{"thought": "你的思考过程", "answer": "最终答案"}}

Question: {question}
History: {history}
"""

class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE_JSON.format(tools=tools_desc, question=question, history=history_str)

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                print("错误：LLM未能返回有效响应。")
                break

            final_answer = self._step(response_text)
            if final_answer is not None:
                return final_answer

        print("已达到最大步数，流程终止。")
        return None

    # __init__ 与 run 的循环骨架保持不变，仅替换解析逻辑
    def _parse_response(self, text: str):
        """从LLM输出中解析JSON，返回 (thought, tool_name, tool_input, final_answer)。"""
        if not text:
            return None, None, None, None
        # 1) 剥离可能的 Markdown 代码围栏
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
        # 2) 截取第一个 { 到最后一个 } 的区间（容忍模型前后夹带文字）
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return None, None, None, None
        raw = cleaned[start:end + 1]
        # 3) 标准 JSON 优先，失败回退 literal_eval（容忍单引号）
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(raw)
            except Exception:
                return None, None, None, None
        if not isinstance(data, dict):
            return None, None, None, None
        thought = data.get("thought")
        if "answer" in data:
            return thought, None, None, data.get("answer")
        action = data.get("action")
        if isinstance(action, dict):
            return thought, action.get("name"), action.get("input"), None
        return thought, None, None, None

    # run 循环内原"解析 + 执行"片段替换为：
    def _step(self, response_text: str):
        thought, tool_name, tool_input, final_answer = self._parse_response(response_text)
        if thought:
            print(f"思考: {thought}")
        if final_answer is not None:
            print(f"最终答案: {final_answer}")
            return final_answer
        if not tool_name:
            # 解析失败：回灌错误信息让模型下一轮自纠，而不是直接终止
            self.history.append("Observation: 解析失败——你上一轮输出不是合法 JSON，或缺少 action/answer 字段，请重新按格式输出。")
            return None
        tool_function = self.tool_executor.getTool(tool_name)
        if not tool_function:
            observation = f"错误: 未找到名为 '{tool_name}' 的工具。可用工具: {self.tool_executor.getAvailableTools()}"
        else:
            observation = tool_function(tool_input)
        print(f"行动: {tool_name}[{tool_input}]")
        print(f"观察: {observation}")
        self.history.append(f"Action: {tool_name}[{tool_input}]")
        self.history.append(f"Observation: {observation}")
        return None

if __name__ == '__main__':
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    calculator_desc = "一个数学计算器。当你需要精确计算算术表达式（如 (123+456)*789/12）时使用此工具，可避免大模型自身计算错误。"
    tool_executor.registerTool("Calculator", calculator_desc, calculator)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)
    question2 = "计算 (123 + 456) × 789 / 12 的结果是多少？"
    agent.run(question2)
