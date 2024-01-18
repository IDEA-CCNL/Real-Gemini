#encoding=utf8

import requests
import argparse

def text_to_music(text: str, host: str, port: int):
    url = f"http://{host}:{port}/text_to_music"
    data = {"text": text}
    response = requests.post(url, data=data)
    return response.content

def tts(text: str, host: str, port: int):
    url = f"http://{host}:{port}/tts"
    data = {"prompt": text}
    response = requests.post(url, data=data)
    return response.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Server API")
    parser.add_argument("--host", type=str, default="localhost", help="Host")
    parser.add_argument("--port", type=int, default=6678, help="Port")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt")
    parser.add_argument("--server", type=str, choices=["music", "tts"], required=True)
    args = parser.parse_args()

    if args.server == "music":
        print(text_to_music(args.prompt, args.host, args.port))
    elif args.server == "tts":
        print(tts(args.prompt, args.host, args.port))
