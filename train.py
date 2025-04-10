import torch
import tiktoken
from model import Model,Config


#Hyperparameters
batch_size = 16
context_length = 16
max_iters = 200        # 循环多少次
learning_rate = 1e-3     # 学习率动态
eval_interval = 20
eval_iters = 5 # 评估次数：在每次评估时，使用验证集中的 10 个批次的数据来计算模型的性能(如损失值或准确率)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)


with open('data/train.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = tiktoken.get_encoding('cl100k_base')
tokenizer_text = tokenizer.encode(text)
tokenizer_text = torch.tensor(data=tokenizer_text,dtype=torch.long,device=device)

p_size = int(len(tokenizer_text) * 0.9)
train_data = tokenizer_text[:p_size]
valid_data = tokenizer_text[p_size:]

config = Config()
model = Model(config).to(device)

def get_batch(split):
    data = train_data if split == 'train' else valid_data
    idxs = torch.randint(low=0,high=len(data)-context_length,size=(batch_size,)) #batch_size个随机数
    x = torch.stack([data[idx:idx+context_length] for idx in idxs]) #[batch_size,context_length]
    y = torch.stack([data[idx+1:idx+context_length+1] for idx in idxs])
    return x, y



@torch.no_grad()
def estimate_loss(): # 打印训练集和验证集的loss
    out = {}
    model.eval() # 验证的时候不更新梯度
    for split in ['train','valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):# 5次循环
            x, y = get_batch(split)
            loss  = model(x, y)["loss"]
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # 训练的时候更新梯度
    return out # 返回一个字典 {'train':losses.mean(),'valid':losses.mean()}




optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)
for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print('Step:',step,'|Train Loss:',round(losses['train'].item(),3),'|Valid Loss:',round(losses['valid'].item(),3),'|')
    x,y = get_batch('train')
    result = model(x,y)
    loss = result['loss']
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


torch.save(model.state_dict(),'model/model.ckpt')
