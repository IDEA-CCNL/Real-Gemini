from fastapi import FastAPI, File, Form, UploadFile
from typing import List
import requests
from PIL import Image
from io import BytesIO
import base64


from real_gemini.gpt4v import GPT4V

gpt4v = GPT4V()
app = FastAPI()



# API端点
@app.post("/main/")
async def main(
    query: str = Form(...),
    images: List[str] = Form(...)
):
    query = "Guess what movie I'm acting out."
    image_paths = ["./test/test_0.png", "./test/test_1.png", "./test/test_2.png", "./test/test_3.png"]
    print(query, image_paths)

    gpt4v = GPT4V()
    result = gpt4v.process_request(query=query, image_paths=image_paths)

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
