#encoding=utf8

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from real_gemini.agent import ReActAgent

if __name__ == '__main__':
    # arguments containing: config_path, ckpt_path, max_length
    args_parser = argparse.ArgumentParser("test agent")
    args_parser.add_argument("--prompt", type=str, help="prompt string", required=True)
    args_parser.add_argument("--image_path_or_dir", type=str, help="image path or dir", required=True)
    args = args_parser.parse_args()

    work_dir = os.path.dirname(os.path.dirname(__file__))
    env_file = os.path.join(work_dir, '.env')
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_file)

    # image dir是过去一段时间截帧的图片目录
    agent = ReActAgent()
    # agent = SimpleRouterAgent(image_dir=args.image_dir)
    output = agent.run(prompt=args.prompt, image_path_or_dir=args.image_path_or_dir)
    print("output:", output)
