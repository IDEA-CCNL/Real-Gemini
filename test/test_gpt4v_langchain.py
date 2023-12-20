#encoding=utf8

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from real_gemini.gpt4v_langchain import GPT4VTool

if __name__ == '__main__':
    # arguments containing: config_path, ckpt_path, max_length
    args_parser = argparse.ArgumentParser("test gpt4v for langchain tools")
    args_parser.add_argument("--prompt", type=str, help="prompt string", required=True)
    args_parser.add_argument("--image_dir", type=str, help="image dir", required=True)
    args = args_parser.parse_args()

    tool = GPT4VTool(image_dir=args.image_dir)
    tool.inference(args.prompt)