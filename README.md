# Real-Gemini

Real-time video understanding and interaction through text, audio, image and video with large multi-modal model.

利用多模态大模型的实时视频理解和交互框架，通过文本、语音、图像和视频和这是世界进行问答和交流。

## 环境配置

首先需要根据`requirements.txt`安装相关的python包。

```bash
pip install -r requirements.txt
```

在前端环境，同时要安装pyaudio。

```bash
sh install pyaudio.sh
```

你需要设置一些环境变量。首先，将`.env.template`复制到同目录下的`.env`。

```bash
cp .env.template .env
```

然后填写对应的环境变量。

## 启动后端模型服务
你可以在有GPU的后端环境中启动服务，例如启动Text2Music的服务。

```bash
sh scripts/start_music_server.sh
```

## 启动前端对话服务

主要实现使用streamlit实现了前端界面。

然后启动前端：

```shell
sh run.sh
```

请注意，若前端环境是MacOS，请不要使用第三方的Terminial启动服务。

## TTS和ASR服务
- ASR
服务调用自内部自建工具（TODO：更新一个服务并开源）
- TTS
见[tts.py](./real_gemini/tts.py)，启动脚本：
```shell
python tts.py
```
启动这些服务需要一些额外的环境和模型：`torch, torchaudio, TTS`，用`pip`安装即可，模型文件路径见py脚本。

## Acknowledgement
- [Fastapi](https://github.com/tiangolo/fastapi)
- [Streamlit](https://github.com/streamlit/streamlit)
- [sagittarius](https://github.com/gregsadetsky/sagittarius)

## 关于我们 About Us

IDEA研究院封神榜团队是中文大模型开源计划[Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)的负责团队，开源包括[二郎神系列](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B)，[太乙系列](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1)，[姜子牙系列](https://huggingface.co/IDEA-CCNL/Ziya2-13B-Chat)等知名模型，并收获了开源社区的广泛使用和支持。

IDEA研究院CCNL技术团队已创建封神榜开源讨论群，我们将在讨论群中不定期更新发布封神榜新模型与系列文章。请扫描微信搜索“fengshenbang-lm”，添加封神空间小助手进群交流！

The IDEA Research Institute Fengshenbang team is the responsible team for the Chinese large model open source project [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM). The open source includes well-known models such as the [Erlang](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B), [Taiyi](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1), and [Ziya](https://huggingface.co/IDEA-CCNL/Ziya2-13B-Chat), and has received widespread use and support from the open source community.

The IDEA Research Institute CCNL technical team has created an open discussion group for Fengshenbang. We will periodically update and release new Fengshenbang models and series of articles in the discussion group. Please scan WeChat and search for "fengshenbang-lm", and add the Fengshen Space Assistant to join the group discussion!



