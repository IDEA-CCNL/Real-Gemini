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
from .tools.controlnet_tool import Image2Pose
from .tools.sam_tool import Segmenting
from .tools.dino_tool import Text2Box
from .tools.imageediting_tool import ImageEditing, Inpainting
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
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history", output_key='output')
        self.tools.extend(load_tools(["dalle-image-generator"]))
        sam = Segmenting(device="cuda")
        self.tools.append(Tool(
                name=sam._name_,
                description=sam._description_,
                func=sam.inference_all,
        ))
        controlnet = Image2Pose(device="cuda")
        self.tools.append(Tool(
                name=controlnet._name_,
                description=controlnet._description_,
                func=controlnet.inference,
        ))
        dino = Text2Box(device="cuda")
        self.tools.append(Tool(
                name=dino._name_,
                description=dino._description_,
                func=dino.inference,
        ))
        inpainting = Inpainting(device="cuda")
        image_editing = ImageEditing(dino, sam, inpainting)
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

        self.agent = initialize_agent(
              tools=self.tools,
              llm=self.llm,
              memory=memory,
              agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
              verbose=True,
        )

    def run(self, prompt: str, image_path_or_dir: str):
        prompt_template = PromptTemplate.from_template(
            "请参考这个{path_or_dir}下的图片，回答我的问题，请注意，图片和问题可能是没有关联的，你需要自己进行判断。\n图片{path_or_dir}：{image}\n问题：{prompt}"
        )
        prompt = prompt_template.format(
            prompt=prompt, image=image_path_or_dir,
            path_or_dir="目录" if os.path.isdir(image_path_or_dir) else "路径")
        output = self.agent.run(prompt)
        print(self.agent.memory.load_memory_variables({}))
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
