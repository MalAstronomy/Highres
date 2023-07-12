import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np

# device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
# batch_size = 32
# block_size = 512
# n_embedding = 512
# n_blocks = 12
# n_head = 8        
dropout = 0.8
# max_iters = 50000
# learning_rate = 3e-4
# eval_every = 500
# eval_iters = 200
# save_every = 10000

import tokenizers
tokenizer = tokenizers.ByteLevelBPETokenizer()
# tokenizer.train("harry-potter.txt")
vocab_size = tokenizer.get_vocab_size()
encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l)

class Head(nn.Module):
    def __init__(self, head_size, n_embedding, block_size):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, H)
        q = self.query(x) # (B, T, H)
        H = k.shape[-1]

        # compute attention scores (affinities)
        wei = q @ k.transpose(-1, -2) * H**-0.5 # (B, T, H) @ (B, H, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # weighted aggregation of the values
        v = self.value(x) # (B, T, H)
        h = wei @ v # (B, T, T) @ (B, T, H) -> (B, T, H)

        return h

class MuliHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, n_embedding, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embedding, block_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embedding, n_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.cat([h(x) for h in self.heads], dim=-1)
        # print(h)
        h = self.proj(h)
        h = self.dropout(h)
        return h

class FeedForward(nn.Module):
    def __init__(self, n_embedding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding, 4 * n_embedding),
            nn.GELU(),
            nn.Linear(4 * n_embedding, n_embedding),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embedding, n_head, block_size):
        super().__init__()
        head_size = n_embedding // n_head
        
        self.sa = MuliHeadAttention(n_head, head_size, n_embedding, block_size)
        self.ff = FeedForward(n_embedding)
        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)

    def forward(self, x):
        h = x       
        h = h + self.sa(self.ln1(h))
        h = h + self.ff(self.ln2(h))
        return h


class GPTLanguageModel(nn.Module):
    def __init__(self, n_embedding, n_head, n_blocks, block_size):
        super().__init__()
        # self.maxpool = nn.AvgPool1d(128, stride=16)
        self.g  = nn.Linear(2, 32) 
        self.e1 = nn.ELU()
        self.f = nn.Linear(32, 128)
        self.e2 = nn.ELU()

        self.e = nn.Flatten()
        self.ln_f = nn.LayerNorm(n_embedding) # final layer norm
        self.l1 = nn.Linear(48256,12000)
        self.l2 = nn.Linear(12000, 6000)
        self.l3 = nn.Linear(6000, 1000)

        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding_table = nn.Embedding(block_size, n_embedding)
        self.blocks = nn.Sequential(*[Block(n_embedding, n_head, block_size) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(n_embedding) # final layer norm
        self.lm_head = nn.Linear(n_embedding, vocab_size)
        
    def forward(self, x): # (B, T)
        # h = x  #(B, 2, 6144) L = 6144
        # print('1', h.size()) 
        # 
        # print('2', h.size())
        x = torch.permute(x, (0, 2, 1)) #(B, 6144, 2)
        print(x.is_cuda, 'a')
        # print('3', h.size())
        x = self.e2(self.f(self.e1(self.g(x)))) #(B, L, 128)
        print(x.is_cuda, 'b', np.shape(x), np.shape(x.long()))

        B, T, C = x.shape
        tok_emb = self.token_embedding_table(x) # (B, T, C)
        print(tok_emb.is_cuda, '1', np.shape(tok_emb))
        pos_emb = self.position_embedding_table(torch.arange(T, device="cuda")) # (T, C)
        print(pos_emb.is_cuda, '2', np.shape(pos_emb))
        x = tok_emb + pos_emb # (B, T, C)
        print(x.is_cuda)
        x = self.blocks(x.double()) # B, 6144, 128
        print(x.is_cuda)
        x = self.ln_f(x) # B, 6144, 128
        print(x.is_cuda)

        x = torch.permute(x, (0, 2, 1)) # B, 128, 6144
        print(x.is_cuda)
        x = self.maxpool(x)  #(B, 128, 377) L' = 377
        print(x.is_cuda)

        x = self.e(x) #(B, 48256)
        print(x.is_cuda)
        x = self.e1(self.l3(self.e1(self.l2(self.e1(self.l1(x)))))) #(B, 1000)
        print(x.is_cuda)

        # logits = self.lm_head(h) # (B, T, V)
        # return logits

        return x

class MultiHeadAttentionwithMLP(nn.Module):
    def __init__(self,n_embedding, n_head, n_blocks, block_size):
        super().__init__()
        
        self.maxpool = nn.AvgPool1d(128, stride=16)
        self.g  = nn.Linear(2, 32) 
        self.e1 = nn.ELU()
        self.f = nn.Linear(32, 128)
        self.e2 = nn.ELU()
        self.blocks = nn.Sequential(*[Block(n_embedding, n_head, block_size) for _ in range(n_blocks)])
        self.e = nn.Flatten()
        self.ln_f = nn.LayerNorm(n_embedding) # final layer norm
        self.l1 = nn.Linear(48256,12000)
        self.l2 = nn.Linear(12000, 6000)
        self.l3 = nn.Linear(6000, 1000)

    def forward(self, x):
        h = x  #(B, 2, 6144) L = 6144
        # print('1', h.size()) 
        h = self.maxpool(h)  #(B, 2, 377) L' = 377
        # print('2', h.size())
        h = torch.permute(h, (0, 2, 1)) #(B, L', 2)
        # print('3', h.size())
        h = self.e2(self.f(self.e1(self.g(h)))) #(B, L', 128)
        # print('4', h.size())
        h = self.blocks(h) #(B, L', 128)
        h = self.ln_f(h)
        h = self.e(h) #(B, 48256)
        h = self.e1(self.l3(self.e1(self.l2(self.e1(self.l1(h)))))) #(B, 1000)
        
        # print('5', h.size())
        # h = self.e(h)    #(B, L', 1)
        # print('6', h.size())
        # h = torch.squeeze(h)   #(B, L')
        # print('7', h.size())

        ## h = h.to(torch.double)

        return h
        
