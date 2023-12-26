#encoding=utf8

import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.tools import Tool
from langchain.agents import initialize_agent, load_tools, AgentType
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

from .tools.gpt4v_tool import GPT4VTool

class SimpleGPT4VAgent(object):
    def __init__(self, image_dir: str):
        self.gpt4v = GPT4VTool(image_dir)

    def run(self, prompt: str):
        output = self.gpt4v.inference(prompt)
        # print(output)
        return output

class ReActAgent(object):

    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
        # self.llm = OpenAI(temperature=0.5)
        gpt4v = GPT4VTool()
        self.tools = [Tool(
              name=gpt4v._name_,
              description=gpt4v._description_,
              func=gpt4v.inference,
        )]
        memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.tools.extend(load_tools(["dalle-image-generator"]))
        self.agent = initialize_agent(
              tools=self.tools,
              llm=self.llm,
              agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
              verbose=True,
        )

    def run(self, prompt: str, image_dir: str):
        prompt_template = PromptTemplate.from_template(
            "请根据这个目录下的图片，回答我的问题。\n图片目录：{image_dir}\n问题：{prompt}"
        )
        prompt = prompt_template.format(prompt=prompt, image_dir=image_dir)
        output = self.agent.run(prompt)
        return output

INTENT_TEMPLATE = """
现在我将给你一个问题，你负责进行问题意图识别，回答这个问题将要调用哪个工具来。目前有以下几个工具：
{all_tool_desc}
请根据[tool description]的描述来判断意图，并仅输出[tool name]对应的值。
现在，我的问题是：
{{question}}
现在，需要调用的工具是：
"""

QUESTION_TEMPLATE = """
你是一个多模态专家。请回答下面这个问题：
{question}
"""

class SimpleRouterAgent(object):
    def __init__(self, image_dir: str):
        self.llm = OpenAI(temperature=0.5)
        gpt4v = GPT4VTool(image_dir)
        self.tools = [Tool(
              name=gpt4v._name_,
              description=gpt4v._description_,
              func=gpt4v.inference,
        )]
        self.tools.extend(load_tools(["dalle-image-generator"]))
        all_tool_desc = ""
        for idx, tool in enumerate(self.tools):
            all_tool_desc += f"##TOOL {idx+1}##\n[tool name] {tool.name}\n[tool description] {tool.description}\n\n"
        intent_template = INTENT_TEMPLATE.format(all_tool_desc=all_tool_desc)
        print(intent_template)
        self.intent_chain = (
            PromptTemplate.from_template(intent_template) | 
            ChatOpenAI(model_name="gpt-4", temperature = 0.3) | 
            StrOutputParser()
        )
        # branch = RunnableBranch(
        #     *[(lambda x: x["intent"].lower() == tool.name, tool.invoke) for tool in self.tools],
        #     self.tools[0].invoke
        # )
        # self.full_chain = {"intent": self.intent_chain, "question": lambda x: x["question"]} | branch
    
    def run(self, prompt: str):
        intent = self.intent_chain.invoke({"question": prompt})
        print("intent", intent)
        for tool in self.tools:
            print(tool.name)
            if intent.lower() == tool.name.lower():
                return tool.invoke(prompt)
        else:
            return self.tools[0].invoke(prompt)
