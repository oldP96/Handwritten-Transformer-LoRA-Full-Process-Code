import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any

# Hyperparameters
class Config:
    D_MODEL = 512  # 语义模型维度
    CONTEXT_LENGTH = 16 # 样本Token长度
    NUM_HEADS = 8       # 多头注意力的头数
    DROPOUT = 0.1       # 丢弃率防止过拟合
    D_KEY = D_MODEL // NUM_HEADS     # 每个头维度
    NUM_BLOCKS = 12                  # Transformer块的数量
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MAX_TOKEN_VALUE = 100256  # 词汇表大小

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

class FeedForwordNetwork(nn.Module):
    """前馈神经网络模块"""
    def __init__(self, config: Config):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(config.D_MODEL, 4 * config.D_MODEL),
            nn.ReLU(),
            nn.Linear(4 * config.D_MODEL, config.D_MODEL),
            nn.Dropout(config.DROPOUT)
        )
    
    # 输入张量，输出张量(X)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
    
class Attention(nn.Module):
    """单头注意力机制"""
    def __init__(self, config: Config):
        super().__init__()
        self.Wq = nn.Linear(config.D_MODEL, config.D_KEY, bias=False)
        self.Wk = nn.Linear(config.D_MODEL, config.D_KEY, bias=False)
        self.Wv = nn.Linear(config.D_MODEL, config.D_KEY, bias=False)
        # mask 切下三角形
        self.register_buffer('mask', torch.tril(torch.ones(config.CONTEXT_LENGTH, config.CONTEXT_LENGTH)))
        self.dropout = nn.Dropout(config.DROPOUT)
        self.scale = math.sqrt(config.D_KEY)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch_size, [Timestep]context_length, d_model)
        B, T, D = x.shape
        q = self.Wq(x)  # (batch_size, context_length, d_key)
        k = self.Wk(x)
        v = self.Wv(x)

        # scale计算注意力分数
        scores = (q @ k.transpose(-2, -1)) / self.scale
        scores = scores.masked_fill_(self.mask[:T, :T] == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)  # Optional
        output = attention_weights @ v
        return output


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, config: Config):
        super().__init__()
        self.heads = nn.ModuleList([Attention(config) for _ in range(config.NUM_HEADS)])
        self.Wo = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, context_length, d_model)
        B, T, D = x.shape
        # Concatenate 将所有注意力头的输出拼接在一起
        outputs = torch.cat([head(x) for head in self.heads], dim=-1)  # (batch_size, context_length, num_heads * d_key)
        # Reshape output to (batch_size, context_length, d_model)
        outputs = outputs.view(B, T, -1)
        # Linear transformation(线性变换后续网络层使用)
        outputs = self.Wo(outputs)  # (batch_size, context_length, d_model)
        outputs = self.dropout(outputs)
        return outputs
    
class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, config: Config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ffn = FeedForwordNetwork(config)
        self.ln1 = nn.LayerNorm(config.D_MODEL)
        self.ln2 = nn.LayerNorm(config.D_MODEL)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class PositionalEncoding(nn.Module):
    """位置编码使用sin和cos函数[-1,1]"""
    def __init__(self, config: Config):
        super().__init__()
        self.d_model = config.D_MODEL
    
    def forward(self, seq_len: int, device: str) -> torch.Tensor:
        pe = torch.zeros(seq_len, self.d_model, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, self.d_model, 2).float() / self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.MAX_TOKEN_VALUE, config.D_MODEL) # token embedding
        self.positional_encoding = PositionalEncoding(config)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.NUM_BLOCKS)])
        # 可以直接使用nn.Linear(config.D_MODEL, config.MAX_TOKEN_VALUE)，我加了一层层归一化(关注度更接近)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)
        self.vocab_linear = nn.Linear(config.D_MODEL, config.MAX_TOKEN_VALUE)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, Any]: # x: (batch_size, [Timestep]context_length)
        B, T = x.shape
        # Embed the input tokens
        x_embed = self.token_embedding(x)  # x_embed: (batch_size, context_length, d_model)
        pe_lookup_table = self.positional_encoding(T, self.config.DEVICE)
        output = x_embed + pe_lookup_table
        output = self.transformer_blocks(output)
        output = self.layer_norm(output)
        logits = self.vocab_linear(output)

        result = {"logits": logits}

        if y is not None:
            B, T, D = logits.shape
            logits_reshape = logits.view(B * T, D)
            y_reshape = y.view(B * T)
            loss = F.cross_entropy(logits_reshape, y_reshape)
            result["loss"] = loss
        else:
            # 当 y 为 None 时，返回一个默认的损失值
            result["loss"] = torch.tensor(0.0, device=self.config.DEVICE)
    
        return result
    
    
    def generate(self, x: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0) -> torch.Tensor:
        # x: (batch_size, [Timestep]context_length)
        for _ in range(max_new_tokens):
            x_crop = x[:, -self.config.CONTEXT_LENGTH:]
            logits = self.forward(x_crop)["logits"]  # logits: (batch_size(1), [Timestep]context_length(1), vocab_size(100256))
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            predicted_token = torch.multinomial(probs, num_samples=1)  # predicted_token
            x = torch.cat((x, predicted_token), dim=1)

        return x