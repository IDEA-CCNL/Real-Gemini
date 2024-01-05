#encoding=utf8

import os
import re
from langchain.chat_models import ChatOpenAI
from langchain.agents.tools import Tool
from langchain.agents import initialize_agent, load_tools, AgentType
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from .tools.gpt4v_tool import GPT4VTool
from .tools.music_tool import Text2MusicTool
from .tools.controlnet_tool import Image2PoseTool
from .tools.sam_tool import SegmentingTool
from .tools.dino_tool import Text2BoxTool
from .tools.imageediting_tool import ImageEditingTool
from .utils.agent_prompt import PREFIX, FORMAT_INSTRUCTIONS, SUFFIX

class ReActAgent(object):

    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
        # self.llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.5)
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
        self.tools.extend(load_tools(["dalle-image-generator"]))
        sam = SegmentingTool()
        self.tools.append(Tool(
                name=sam._name_,
                description=sam._description_,
                func=sam.inference,
        ))
        controlnet = Image2PoseTool()
        self.tools.append(Tool(
                name=controlnet._name_,
                description=controlnet._description_,
                func=controlnet.inference,
        ))
        dino = Text2BoxTool()
        self.tools.append(Tool(
                name=dino._name_,
                description=dino._description_,
                func=dino.inference,
        ))
        image_editing = ImageEditingTool()
        self.tools.append(Tool(
                name=image_editing._remove_name_,
                description=image_editing._remove_description_,
                func=image_editing.inference_remove,
        ))
        self.tools.append(Tool(
                name=image_editing._replace_name_, 
                description=image_editing._replace_description_,
                func=image_editing.inference_replace_sam,
        ))

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
            "这个{path_or_dir}下的图片是AI多模态助手的关键帧图片，请根据这些图片回答我的问题，请注意，图片和问题可能是没有关联的，你需要自己进行判断。\n图片{path_or_dir}：{image}\n问题：{prompt}"
        )
        prompt = prompt_template.format(
            prompt=prompt, image=image_path_or_dir,
            path_or_dir="目录" if os.path.isdir(image_path_or_dir) else "路径")
        output = self.agent.run(prompt)
        # print(self.agent.memory.load_memory_variables({}))
        output_dict = {"text": output}
        if "https://oaidalleapiprodscus.blob.core.windows.net" in output:
            links = self._find_md_links(output)
            url = list(links.values())[0]
            output_dict["image"] = url
            output_dict["text"] = output.replace(url, "")
        audio_path_re = re.compile(r"/.+/Real-Gemini/test/outputs/.+\.wav")
        if audio_path_re.search(output):
            audio_path = audio_path_re.search(output).group()
            output_dict["audio"] = audio_path
            output_dict["text"] = output.replace(audio_path, "")
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
