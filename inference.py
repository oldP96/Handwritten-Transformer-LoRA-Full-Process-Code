import torch
import tiktoken
from model import Model,Config

#Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337 # 随机种子
torch.manual_seed(TORCH_SEED) # 设置随机种子

config = Config()
model = Model(config)
checkpoint = torch.load('model/model.ckpt')
model.load_state_dict(checkpoint)
model.eval() # 模型设置为评估模式就是推理，训练不用这个函数
model.to(device)

tokenizer = tiktoken.get_encoding("cl100k_base")

start = "儒家思想在王阳明中如何体现的？"
start_ids = tokenizer.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long,device=device)[None,...])


with torch.no_grad():
    y = model.generate(x,max_new_tokens=100,temperature=1.0)
    output_text = tokenizer.decode(y[0].tolist())
    # 加入特殊标记
    special_token = "[START_OF_OUTPUT]"
    special_token_end = "[END_OF_OUTPUT]"
    print('-----------')
    print(f"问题是：{start} 然后答案是：{special_token}{output_text}{special_token_end}")
    print('-----------')