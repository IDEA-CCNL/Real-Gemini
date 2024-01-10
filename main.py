import os
import sys

sys.path.append(os.path.dirname(__file__))

if __name__ == '__main__':
    work_dir = os.path.dirname(__file__)
    env_file = os.path.join(work_dir, '.env')
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_file)
    from real_gemini.chat import launch
    launch()
