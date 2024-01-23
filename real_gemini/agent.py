#encoding=utf8

import os
import re
import json
from langchain.chat_models import ChatOpenAI
from langchain.agents.tools import Tool
from langchain.agents import initialize_agent, load_tools, AgentType
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from .tools.gpt4v_tool import GPT4VTool
from .tools.image_generation_tool import TaiyiGeneralTool
from .tools.music_tool import Text2MusicTool
from .tools.controlnet_tool import Image2PoseTool
from .tools.sam_tool import SegmentingTool
from .tools.dino_tool import Text2BoxTool
from .tools.imageediting_tool import ImageRemoveTool, ImageReplaceTool
from .tools.weather_tool import WeatherTool
from .tools.qwen4v_tool import QWEN4VTool
from .utils.output_parser import ConvoOutputParser
from .utils.agent_prompt import PREFIX, FORMAT_INSTRUCTIONS, SUFFIX

REGISTERED_TOOL_CLASSES = [
    GPT4VTool,
    TaiyiGeneralTool,
    Text2MusicTool,
    SegmentingTool,
    Image2PoseTool,
    Text2BoxTool,
    ImageRemoveTool,
    ImageReplaceTool,
    WeatherTool,
    QWEN4VTool
]

class ReActAgent(object):

    def __init__(self):
        # self.llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
        self.llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.5)
        # self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
        # self.llm = ChatOpenAI(model_name="deepseek", temperature=0.5, openai_api_base="http://192.168.81.13:6767/v1/")
        # self.tools = load_tools(["dalle-image-generator"])
        self.tools = []
        for tool_cls in REGISTERED_TOOL_CLASSES:
            custom_tool = tool_cls()
            self.tools.append(
                Tool(
                    name=custom_tool._name_,
                    description=custom_tool._description_,
                    func=custom_tool.inference,
                    return_direct=custom_tool._return_direct_
                )
            )

        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history", output_key='output')
        
        ai_prefix = "REAL-GEMINI"
        output_parser = ConvoOutputParser()
        output_parser.ai_prefix = ai_prefix
        
        self.agent = initialize_agent(
              tools=self.tools,
              llm=self.llm,
              memory=memory,
              agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
              agent_kwargs={
                'prefix': PREFIX,
                'suffix': SUFFIX,
                'format_instructions': FORMAT_INSTRUCTIONS,
                'ai_prefix': ai_prefix,
                'output_parser': output_parser,
              },
              verbose=True,
        )

    def run(self, prompt: str, image_path_or_dir: str):
        prompt_template = PromptTemplate.from_template(
            "图片{path_or_dir}：{image}\n问题：{prompt}。"
        )
        prompt = prompt_template.format(
            prompt=prompt, image=image_path_or_dir,
            path_or_dir="目录" if os.path.isdir(image_path_or_dir) else "路径")
        output = self.agent.run(prompt)
        # print(self.agent.memory.load_memory_variables({}))
        print("output:", output)
        try:
            output_dict = json.loads(output)
        except:
            output_dict = {"text": output}
        
        return output_dict

