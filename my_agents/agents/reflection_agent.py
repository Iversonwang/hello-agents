from typing import Dict, Optional
from hello_agents import ReflectionAgent
from my_agents.core.config import Config
from my_agents.core.llm import MyLLM

DEFAULT_PROMPTS = {
    "initial": """
    请根据以下要求完成任务:

    任务: {task}

    请提供一个完整、准确的回答。
    """,
    "reflect": """
    请仔细审查以下回答，并找出可能的问题或改进空间:

    # 原始任务:
    {task}

    # 当前回答:
    {content}

    请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
    如果回答已经很好，请回答"无需改进"。
    """,
    "refine": """
    请根据反馈意见改进你的回答:

    # 原始任务:
    {task}

    # 上一轮回答:
    {last_attempt}

    # 反馈意见:
    {feedback}

    请提供一个改进后的回答。
    """,
}


class MyReflectionAgent(ReflectionAgent):
    """
    自定义的反思Agent，继承自ReflectionAgent
    """

    def __init__(
        self,
        name: str,
        llm: MyLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations
        self.prompts = custom_prompts if custom_prompts else DEFAULT_PROMPTS
        print(f"✅ {name} 初始化完成，反思功能已启用")

    def run(self, task: str, **kwargs) -> str:
        """
        重写的运行方法 - 实现反思逻辑
        """
        print(f"🤖 {self.name} 正在处理任务: {task}")

        # 初始回答
        initial_response = self._generate_initial_response(task, **kwargs)
        print(f"初始回答: {initial_response}")

        # 反思阶段
        feedback = self._reflect_on_response(task, initial_response, **kwargs)
        print(f"反馈意见: {feedback}")

        # 改进阶段
        refined_response = self._refine_response(
            task, initial_response, feedback, **kwargs
        )
        print(f"改进后的回答: {refined_response}")

        return refined_response

    def _generate_initial_response(self, task: str, **kwargs) -> str:
        """生成初始回答"""
        prompt = self.prompts.get("initial", DEFAULT_PROMPTS["initial"])
        formatted_prompt = prompt.format(task=task)
        return self.llm.invoke(
            [{"role": "system", "content": formatted_prompt}], **kwargs
        )

    def _reflect_on_response(self, task: str, content: str, **kwargs) -> str:
        """对初始回答进行反思"""
        prompt = self.prompts.get("reflect", DEFAULT_PROMPTS["reflect"])
        formatted_prompt = prompt.format(task=task, content=content)
        return self.llm.invoke(
            [{"role": "system", "content": formatted_prompt}], **kwargs
        )

    def _refine_response(
        self, task: str, last_attempt: str, feedback: str, **kwargs
    ) -> str:
        """根据反馈改进回答"""
        prompt = self.prompts.get("refine", DEFAULT_PROMPTS["refine"])
        formatted_prompt = prompt.format(
            task=task, last_attempt=last_attempt, feedback=feedback
        )
        return self.llm.invoke(
            [{"role": "system", "content": formatted_prompt}], **kwargs
        )
