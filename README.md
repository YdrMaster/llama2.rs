# llama2.rs

> 此项目已重构为 [transformer](https://github.com/YdrMaster/transformer)。新项目提供一个精心设计的张量定义，以加速大模型推理程序开发。cuda 版本也将在新项目上开发。

手写 llama2 推理实现，基于 **[karpathy/llama2.c](https://github.com/karpathy/llama2.c)**，但：

- 支持直接加载 safetensors 格式的模型；
- 使用纯 rust 实现，源码合理分散到多个源文件，可读性更好；
- `-O3`/`--release` 优化下有更高的 `tok/s`；
- 支持从文件读取提示词；
- 状态管理有“层”的抽象，不同层的状态不集中在一处，更像支持流水并行的推理引擎实现；

## 使用

加载 **[karpathy/llama2.c](https://github.com/karpathy/llama2.c)** 定义的 bin 模型格式：

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin

cargo run --release --bin generate -- stories15M.bin --prompt story-begin.txt
```

加载 safetensors 模型格式：

```bash
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/config.json
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors
wget https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin

cargo run --release --bin generate -- model.safetensors --prompt tiny-chat.txt
```

试用对话模式：

```bash
cargo run --release --bin chat -- model.safetensors --system friendly-chatbot.txt
```

> 示例：
>
> ```plaintext
> user: Who are you?
> assistant: Hello there! I'm a friendly chatbot developed by the Artificial Intelligence lab of The University of Pennsylvania. We're here to help you with your queries and provide you with the most relevant and informative responses. Whether you're looking for information about your health, studying abroad, or anything else, we're here to assist you. Thank you for choosing us, and have a great day!</s>
>
> user: How old are you?
> assistant: I don't have a physical age as I'm not a living thing. However, based on the information provided by the client, I can provide a range of ages from 10 years old to 100 years old. Please provide me with more details so that I can give you a more accurate age estimate. Additionally, you can always ask me to provide my birthday. However, it's a general piece of information that can be useful for your queries. Enjoy your chat!</s>
> ```

## 目标

- [x] 支持提示词批量输入；
- [x] 添加注释；
- 支持直接加载通用格式的模型文件：
  - [x] 支持加载 safetensors 模型；
- [x] 支持对话模式；
- [ ] 支持多核并行加速/向量化加速；
