# Transformer(MHA)-LoRA-whole-process
## 作者
depeng wang
## 联系我
wangdepeng97@163.com
## 项目简介
本次项目采用 PyTorch 手写 Transformer 的代码，训练模型并通过 LoRA（Low-Rank Adaptation）进行微调。代码注释完整，适合学习和参考。虽然训练循环次数较短，模型效果尚未达到最佳，但可以作为研究和实践的参考。
<strong style="font-weight: 900;">本项目完全可以拿过来企业应用，但是没有做GPU相关的如混合精度训练，没有设计分布式GPU部署相关</strong>
本项目未使用 HuggingFace 的 Transformer 类库，仅使用了 PEFT 的 LoRA 类库。模型基于 Transformer 的 decoder 实现，尤其是多头注意力机制，逻辑简单清晰。
# 自己clone下来，在vscode中同级目录创建data&model文件夹即可拿自己的样本训练微调
### Project Thinking
The current models are all based on Transformer parameter tuning. The technology beyond Transformer is only "Dao" in China.
As far as I know, no one can explain the neural network, including the academic community, and it is not clear how to operate, so I say that only China's "Dao" can really solve this problem.
Technologies beyond transformer are often the simplest theory in Chinese RU SHI DAO

## 项目特点
- **新增GRPO思维链(参考学习为主)**:使用RL奖励函数，实现基座模型思维过程训练样本级
- **手写 Transformer**：完全从零实现 Transformer 的 decoder，特别是多头注意力机制。
- **LoRA 微调**：采用 LoRA 技术进行模型微调，适合资源有限的场景。
- **学习率策略**：采用 CosineAnnealingLR 学习率衰减策略，模拟学习过程中的“先快后慢”特性。
- **优化器**：使用 AdamW 优化器，提升训练效果。
- **可视化工具**：支持 TensorBoard 和 wandb（推荐 wandb）进行训练梯度可视化。

## 环境依赖
在运行项目前，请确保安装以下依赖包：
```bash
pip install torch torchvision torchaudio
pip install peft
pip install tensorboard
pip install wandb  # 推荐安装
