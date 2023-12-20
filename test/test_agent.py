#encoding=utf8

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

os.environ["OPENAI_API_KEY"] = "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"

from real_gemini.agent import RealGeminiAgent

if __name__ == '__main__':
    # arguments containing: config_path, ckpt_path, max_length
    args_parser = argparse.ArgumentParser("test gpt4v for langchain tools")
    args_parser.add_argument("--prompt", type=str, help="prompt string", required=True)
    args_parser.add_argument("--image_dir", type=str, help="image dir", required=True)
    args = args_parser.parse_args()

    # image dir是过去一段时间截帧的图片目录
    agent = RealGeminiAgent(image_dir=args.image_dir)
    output = agent.run(args.prompt)
    print("output:", output)
