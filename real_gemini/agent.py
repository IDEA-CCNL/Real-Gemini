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
# from .tools.weather_tool import WeatherTool
from .tools.music_tool import Text2MusicTool

class ReActAgent(object):

    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
        # self.llm = OpenAI(temperature=0.5)
        gpt4v = GPT4VTool()
        # weather_tool = WeatherTool()
        music_tool = Text2MusicTool()
        self.tools = [
            Tool(
              name=gpt4v._name_,
              description=gpt4v._description_,
              func=gpt4v.inference,
            ),
            # Tool(
            #     name=weather_tool._name_,
            #     description=weather_tool._description_,
            #     func=weather_tool.inference,
            # ),
            Tool(
                name=music_tool._name_,
                description=music_tool._description_,
                func=music_tool.inference,
            )
        ]
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history", output_key='output')
        self.tools.extend(load_tools(["dalle-image-generator"]))
        self.agent = initialize_agent(
              tools=self.tools,
              llm=self.llm,
              memory=memory,
              agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
              verbose=False,
        )

    def run(self, prompt: str, image_path_or_dir: str):
        prompt_template = PromptTemplate.from_template(
            "请参考这个{path_or_dir}下的图片，回答我的问题，请注意，图片和问题可能是没有关联的，你需要自己进行判断。\n图片{path_or_dir}：{image}\n问题：{prompt}"
        )
        prompt = prompt_template.format(
            prompt=prompt, image=image_path_or_dir,
            path_or_dir="目录" if os.path.isdir(image_path_or_dir) else "路径")
        output = self.agent.run(prompt)
        # print(self.agent.memory.load_memory_variables({}))
        return output
