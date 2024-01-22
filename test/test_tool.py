#encoding=utf8

import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from real_gemini.tools.music_tool import Text2MusicTool
from real_gemini.tools.image_generation_tool import TaiyiGeneralTool
from real_gemini.tools.weather_tool import WeatherTool
# from real_gemini.tools.tts_tool import TTSTool
from real_gemini.tools.qwen4v_tool import QWEN4VTool

TOOL_DICT = {
    "music": Text2MusicTool,
    "image": TaiyiGeneralTool,
    "weather": WeatherTool,
    # "tts": TTSTool,
    "qwen4v": QWEN4VTool
}

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser("test tool")
    args_parser.add_argument("--prompt", type=str, help="prompt string", required=True)
    args_parser.add_argument("--tool", type=str, help="tool name", required=True, choices=["music", "image", "weather", "tts", "qwen4v"])
    args = args_parser.parse_args()

    work_dir = os.path.dirname(os.path.dirname(__file__))
    env_file = os.path.join(work_dir, '.env')
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_file)

    tool = TOOL_DICT[args.tool]()

    # prompt = {"question":"这是正确的顺序吗？考虑与太阳的距离并解释你的推理。", \
    #           "image_input":os.path.join(work_dir, "test/images/reasoning_logic/reasoning_logic_1.png")}
    # args.prompt = json.dumps(prompt)

    resp = tool.inference(args.prompt)
    print("resp:", resp)

    