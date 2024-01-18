# llama2.rs

手写 llama2 推理实现，基于 **[karpathy/llama2.c](https://github.com/karpathy/llama2.c)**，但：

- 使用纯 rust 实现，源码合理分散到多个源文件，可读性更好；
- `-O3`/`--release` 优化下有更高的 `tok/s`；
- 支持从文件读取提示词；
- 状态管理有“层”的抽象，不同层的状态不集中在一处，更像支持流水并行的推理引擎实现；

## 使用

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin

cargo run --release --bin generate -- stories15M.bin --prompt story-begin.txt
```

## 目标

- [x] 支持提示词批量输入；
- [x] 添加注释；
- 支持直接加载通用格式的模型文件：
  - [ ] 支持加载 safetensors 模型；
- [ ] 支持问答模式；
- [ ] 支持多核并行加速/向量化加速；
