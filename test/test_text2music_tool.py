#encoding=utf8

import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from real_gemini.tools.music_tool import Text2MusicTool

if __name__ == '__main__':
    # arguments containing: config_path, ckpt_path, max_length
    args_parser = argparse.ArgumentParser("test music tool")
    args_parser.add_argument("--prompt", type=str, help="prompt string", required=True)
    args = args_parser.parse_args()

    work_dir = os.path.dirname(os.path.dirname(__file__))
    env_file = os.path.join(work_dir, '.env')
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_file)

    tool = Text2MusicTool()
    resp = tool.inference(args.prompt)
    print("resp:", resp)
