#encoding=utf8

import requests
import argparse

def text_to_music(text: str, host: str, port: int):
    url = f"http://{host}:{port}/text_to_music"
    data = {"text": text}
    response = requests.post(url, data=data)
    return response.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio API")
    parser.add_argument("--host", type=str, default="localhost", help="Host")
    parser.add_argument("--port", type=int, default=6678, help="Port")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt")
    args = parser.parse_args()

    print(text_to_music(args.prompt, args.host, args.port))
