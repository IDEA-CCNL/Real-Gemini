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
]

class ReActAgent(object):

    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
        # self.llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.5)
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
        
        self.agent = initialize_agent(
              tools=self.tools,
              llm=self.llm,
              memory=memory,
              agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
              prefix=PREFIX,
              suffix=SUFFIX,
              format_instructions=FORMAT_INSTRUCTIONS,
              verbose=True,
        )

    def run(self, prompt: str, image_path_or_dir: str):
        prompt_template = PromptTemplate.from_template(
            "这个{path_or_dir}下的图片是AI多模态助手的截取的关键帧图片，请根据这些图片回答我的问题，请注意，图片和问题可能是没有关联的，你需要自己进行判断。\n图片{path_or_dir}：{image}\n问题：{prompt}。\n\n由于你输入的文字是由ASR服务得到的，所以可能存在一些识别噪音。假如你的输入是一些没有意义的文字或者不通顺的句子时，请不要调用工具，并直接返回\"###\"。"
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
        if "https://oaidalleapiprodscus.blob.core.windows.net" in output:
            links = self._find_md_links(output)
            url = list(links.values())[0]
            output_dict["image"] = url
            output_dict["text"] = output.replace(url, "")
        
        return output_dict

    def _find_md_links(self, md):
        """ Return dict of links in markdown """

        INLINE_LINK_RE = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        FOOTNOTE_LINK_TEXT_RE = re.compile(r'\[([^\]]+)\]\[(\d+)\]')
        FOOTNOTE_LINK_URL_RE = re.compile(r'\[(\d+)\]:\s+(\S+)')

        links = dict(INLINE_LINK_RE.findall(md))
        footnote_links = dict(FOOTNOTE_LINK_TEXT_RE.findall(md))
        footnote_urls = dict(FOOTNOTE_LINK_URL_RE.findall(md))

        for key, value in footnote_links.items():
            footnote_links[key] = footnote_urls[value]
        links.update(footnote_links)
        print(links)
        return links
