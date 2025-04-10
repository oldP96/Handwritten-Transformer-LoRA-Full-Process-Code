# 微调本地模型，使用LoRA进行微调，N*D = N*r @ r*D r={4,8,16,32}
# WLoRA = N*D + N*r(B)全零初始化 @ r*D(A)随机数初始化
# QLoRA B@A 从float32转为int8/4节省GPU 是对预训练模型的参数进行量化
# 微调的vocab_size和训练的基座模型vocab_size一致
# 微调有很多种指令微调，全量微调，LoRA微调等
# 没有使用huggingface的类库模型微调，只是使用LoRA的类库
import torch
from model import Model,Config
from peft import get_peft_model, LoraConfig
import tiktoken
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# 超参数
batch_size = 16      # 批次大小随机取8次16个token
device = 'cuda' if torch.cuda.is_available() else 'cpu'
context_length = 16 # 上下文长度(我爱吃香蕉这个样本有16个token)
max_iters = 100     # 最大训练次数
epochs = 1000   # 训练轮数
eval_interval = 10 # 评估间隔：表示每隔 10 个训练步骤（或批次）进行一次模型评估
learning_rate = 1e-4 # 学习率
TORCH_SEED = 1337 # 随机种子
torch.manual_seed(TORCH_SEED) # 设置随机种子

# 数据预处理
class TextDataset(Dataset):
    def __init__(self, data, context_length):
        self.data = data
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.context_length]
        y = self.data[idx + 1:idx + self.context_length + 1]
        return x, y

# 准备数据集
with open('data/lora.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# 加载数据集并拆分为训练集百分之90和验证集百分之10
tokenizer = tiktoken.get_encoding('cl100k_base')
tokenizer_text = tokenizer.encode(text)
tokenizer_text = torch.tensor(data=tokenizer_text,dtype=torch.long,device=device)

p_size = int(len(tokenizer_text) * 0.9)
train_data = tokenizer_text[:p_size]
valid_data = tokenizer_text[p_size:]

train_dataset = TextDataset(train_data, context_length)
valid_dataset = TextDataset(valid_data, context_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# 加载预训练模型
config = Config()
model = Model(config)
pretrained_state_dict = torch.load("model/model.ckpt")
model_state_dict = model.state_dict()

# 检查权重是否匹配
for key in pretrained_state_dict:
    if key in model_state_dict and pretrained_state_dict[key].shape == model_state_dict[key].shape:
        model_state_dict[key] = pretrained_state_dict[key]
model.load_state_dict(model_state_dict)
model.to(device)


# 配置LoRA微调参数
target_modules = ["Wq", "Wv"] # 注意不同的基座模型不同，目标模块也不同
lora_config = LoraConfig(
    r=4,  # Lora秩
    lora_alpha=32,  # Lora缩放因子
    target_modules=target_modules, 
    lora_dropout=0.05,  # Lora dropout率
    bias="none",  # 是否对偏置进行微调
)

# 使用LoRA微调模型
model = get_peft_model(model, lora_config)

# 训练配置
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# 在比较train.py中添加了动态学习率，学习率会从一个初始值开始，逐渐下降到一个最小值，然后再稍微回升，如此循环
# max_iters * 1.1防止学习率过快下降
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters * 1.1)
writer = SummaryWriter()

# 评估函数
@torch.no_grad()
def estimate_loss(): # 打印训练集和验证集的loss
    model.eval() # 验证的时候不更新梯度(权重)
    out = {}
    for split, loader in [('train', train_loader),('valid',valid_loader)]:
        losses = []
        for batch in loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            loss  = model(x, y)["loss"]
            losses.append(loss.item())
        out[split] = torch.tensor(losses).mean().item()
    model.train() # 训练的时候更新梯度
    return out

# 训练循环
print(f"Total training steps: {max_iters}")
global_step = 0  # 全局步数计数器
for epoch in range(epochs):
    if global_step >= max_iters:
        break
    print(f"Epoch {epoch+1} (Global Step {global_step}/{max_iters})")
    for batch in train_loader:
        if global_step >= max_iters:
            break
        if global_step % eval_interval == 0 or global_step == max_iters - 1:
            losses = estimate_loss()
            print(f"Global Step {global_step}: Train Loss {losses['train']:.4f}, Valid Loss {losses['valid']:.4f}")
            writer.add_scalar('Loss/train', losses['train'], global_step)
            writer.add_scalar('Loss/valid', losses['valid'], global_step)

        x, y = batch
        x, y = x.to(device), y.to(device)
        result = model(x, y)
        loss = result["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        global_step += 1  # 全局计数器递增

writer.close()

# 合并权重
model = model.merge_and_unload()

# 保存微调模型
torch.save(model.state_dict(), "model/finetune_model.ckpt")

# 验证模型加载
try:
    loaded_model = Model(config)
    loaded_model.load_state_dict(torch.load("model/finetune_model.ckpt"))
    loaded_model.to(device)
    print("模型加载成功，微调完成！")
except Exception as e:
    print(f"模型加载失败: {e}")