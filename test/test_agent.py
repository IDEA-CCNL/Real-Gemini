#encoding=utf8

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

os.environ["OPENAI_API_KEY"] = "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"

from real_gemini.agent import ReActAgent
from real_gemini.agent import SimpleRouterAgent

if __name__ == '__main__':
    # arguments containing: config_path, ckpt_path, max_length
    args_parser = argparse.ArgumentParser("test gpt4v for langchain tools")
    args_parser.add_argument("--prompt", type=str, help="prompt string", required=True)
    args_parser.add_argument("--image_path_or_dir", type=str, help="image path or dir", required=True)
    args = args_parser.parse_args()

    # image dir是过去一段时间截帧的图片目录
    agent = ReActAgent()
    # agent = SimpleRouterAgent(image_dir=args.image_dir)
    output = agent.run(prompt=args.prompt, image_path_or_dir=args.image_path_or_dir)
    print("output:", output)
