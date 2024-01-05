import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from real_gemini.chat import launch

if __name__ == '__main__':
    launch()
