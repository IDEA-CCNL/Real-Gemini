#encoding=utf8

import os
from langchain.llms import OpenAI
# from langchain.agents.tools import Tool
# from langchain.agents import initialize_agent, AgentType

from .gpt4v_langchain import GPT4VTool

class RealGeminiAgent(object):
    def __init__(self, image_dir: str):
        self.llm = OpenAI(temperature=0.9)
        self.gpt4v = GPT4VTool(image_dir)
        # self.tools = [Tool(
        #       name=gpt4v._name_,
        #       description=gpt4v._description_,
        #       func=gpt4v.inference,
        # )]
        # self.agent = initialize_agent(
        #       tools=self.tools,
        #       llm=self.llm,
        #       agent=AgentType.OPENAI_FUNCTIONS,
        #       verbose=True,
        # )

    def run(self, prompt: str):
        # output = self.agent.run(prompt)
        output = self.gpt4v.inference(prompt)
        return output
