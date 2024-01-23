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

然后填写对应的环境变量。其中，请注意以下的环境变量：
- `GAODE_API_KEY`：请在[高德开放平台](https://lbs.amap.com/)进行申请；
- `TTS_MODEL_PATH`：请下载[XTTS-v2](https://huggingface.co/coqui/XTTS-v2)；
- `MUSIC_MODEL_PATH`：请下载[musicgen](https://huggingface.co/facebook/musicgen-small)；
- `QWEN_VL_MODEL_PATH`：请下载[Qwen-VL](https://huggingface.co/Qwen/Qwen-VL-Chat)；

## 启动后端模型服务
你可以在有GPU的后端环境中启动服务，例如启动TTS的服务。

```bash
sh scripts/start_tts_server.sh
```

必须要启动的服务有：
- TTS

可选启动的服务有：
- Music：当你需要生成音乐时；
- Qwen：当你需要使用Qwen-VL替换GPT-4V时；

## 启动前端对话服务

主要实现使用streamlit实现了前端界面。

然后启动前端：

```shell
sh run.sh
```

请注意，若前端环境是MacOS，请不要使用第三方的Terminial启动服务。

## ASR服务
- ASR
服务调用自内部自建工具（TODO：更新一个服务并开源）

## Acknowledgement
- [Fastapi](https://github.com/tiangolo/fastapi)
- [Streamlit](https://github.com/streamlit/streamlit)
- [sagittarius](https://github.com/gregsadetsky/sagittarius)

## 关于我们 About Us

IDEA研究院封神榜团队是中文大模型开源计划[Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)的负责团队，开源包括[二郎神系列](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B)，[太乙系列](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1)，[姜子牙系列](https://huggingface.co/IDEA-CCNL/Ziya2-13B-Chat)等知名模型，并收获了开源社区的广泛使用和支持。

IDEA研究院CCNL技术团队已创建封神榜开源讨论群，我们将在讨论群中不定期更新发布封神榜新模型与系列文章。请扫描微信搜索“fengshenbang-lm”，添加封神空间小助手进群交流！

The IDEA Research Institute Fengshenbang team is the responsible team for the Chinese large model open source project [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM). The open source includes well-known models such as the [Erlang](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B), [Taiyi](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1), and [Ziya](https://huggingface.co/IDEA-CCNL/Ziya2-13B-Chat), and has received widespread use and support from the open source community.

The IDEA Research Institute CCNL technical team has created an open discussion group for Fengshenbang. We will periodically update and release new Fengshenbang models and series of articles in the discussion group. Please scan WeChat and search for "fengshenbang-lm", and add the Fengshen Space Assistant to join the group discussion!



