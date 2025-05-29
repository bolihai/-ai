import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 实现位置编码机制，解决自主意力机制的局限性
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=5000):
    super(PositionalEncoding, self).__init__()
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)
        
  def forward(self, x):
    # x: [batch_size, seq_len, d_model]
    return x + self.pe[:x.size(1), :]

# 并行计算多个注意力头，捕捉不同子空间的特征，增强模型表达能力
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, nhead, dropout=0.1):
    super(MultiHeadAttention, self).__init__()
    self.d_model = d_model
    self.nhead = nhead
    self.d_k = d_model // nhead
    
    self.q_linear = nn.Linear(d_model, d_model)
    self.k_linear = nn.Linear(d_model, d_model)
    self.v_linear = nn.Linear(d_model, d_model)
    self.out = nn.Linear(d_model, d_model)
    
    self.dropout = nn.Dropout(dropout)
    self.scale = math.sqrt(self.d_k)
        
  def forward(self, q, k, v, mask=None):
    batch_size = q.size(0)
    
    # 线性投影
    q = self.q_linear(q).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
    k = self.k_linear(k).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
    v = self.v_linear(v).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
    
    # 计算注意力得分
    scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # 应用softmax获取注意力权重
    attn = self.dropout(F.softmax(scores, dim=-1))
    
    # 加权求和
    context = torch.matmul(attn, v)
    context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    
    # 输出线性层
    output = self.out(context)
    return output, attn

# 对注意力输出进行非线性变换，增加模型容量（通常为d_model → 4d_model → d_model）
class PositionwiseFeedForward(nn.Module):
  def __init__(self, d_model, d_ff, dropout=0.1):
    super(PositionwiseFeedForward, self).__init__()
    self.w_1 = nn.Linear(d_model, d_ff)
    self.w_2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(dropout)
      
  def forward(self, x):
    return self.w_2(self.dropout(F.relu(self.w_1(x))))

# 对每个样本的特征维度（最后一维）归一化，稳定训练（对比 BatchNorm，更适合序列数据）
class LayerNorm(nn.Module):
  def __init__(self, features, eps=1e-6):
    super(LayerNorm, self).__init__()
    self.a_2 = nn.Parameter(torch.ones(features))
    self.b_2 = nn.Parameter(torch.zeros(features))
    self.eps = eps
        
  def forward(self, x):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# 结合残差连接（x + sublayer(x)）和层归一化，形成 Transformer 的标准子层结构（如 “归一化→子层→残差”）
class SublayerConnection(nn.Module):
  def __init__(self, size, dropout):
    super(SublayerConnection, self).__init__()
    self.norm = LayerNorm(size)
    self.dropout = nn.Dropout(dropout)
      
  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x)))

# 编码器的单层，包含自注意力（self_attn）和前馈网络（feed_forward），通过子层连接组合
class EncoderLayer(nn.Module):
  def __init__(self, d_model, nhead, d_ff, dropout):
    super(EncoderLayer, self).__init__()
    self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    self.sublayer1 = SublayerConnection(d_model, dropout)
    self.sublayer2 = SublayerConnection(d_model, dropout)
        
  def forward(self, x, mask):
    x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask)[0])
    return self.sublayer2(x, self.feed_forward)

# 解码器的单层，包含自注意力（带前瞻掩码，屏蔽未来位置）、交叉注意力（关注编码器输出）和前馈网络，通过子层连接组合
class DecoderLayer(nn.Module):
  def __init__(self, d_model, nhead, d_ff, dropout):
    super(DecoderLayer, self).__init__()
    self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
    self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    self.sublayer1 = SublayerConnection(d_model, dropout)
    self.sublayer2 = SublayerConnection(d_model, dropout)
    self.sublayer3 = SublayerConnection(d_model, dropout)
      
  def forward(self, x, memory, src_mask, tgt_mask):
    x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask)[0])
    x = self.sublayer2(x, lambda x: self.cross_attn(x, memory, memory, src_mask)[0])
    return self.sublayer3(x, self.feed_forward)

# 堆叠多个EncoderLayer（如nlayers层），形成完整的编码器，最后应用层归一化
class Encoder(nn.Module):
  def __init__(self, layer, nlayers):
    super(Encoder, self).__init__()
    self.layers = nn.ModuleList([layer for _ in range(nlayers)])
    self.norm = LayerNorm(layer.d_model)
      
  def forward(self, x, mask):
    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)

# 堆叠多个DecoderLayer（如nlayers层），形成完整的解码器，最后应用层归一化
class Decoder(nn.Module):
  def __init__(self, layer, nlayers):
    super(Decoder, self).__init__()
    self.layers = nn.ModuleList([layer for _ in range(nlayers)])
    self.norm = LayerNorm(layer.d_model)
      
  def forward(self, x, memory, src_mask, tgt_mask):
    for layer in self.layers:
      x = layer(x, memory, src_mask, tgt_mask)
    return self.norm(x)

# 整合编码器、解码器、词嵌入、位置编码和输出层，形成完整的端到端模型
class Transformer(nn.Module):
  def __init__(self, src_vocab, tgt_vocab, d_model, nhead, nlayers, d_ff, max_len=5000, dropout=0.1):
    super(Transformer, self).__init__()
    
    # 编码器部分
    self.src_embed = nn.Embedding(src_vocab, d_model)
    self.src_pos = PositionalEncoding(d_model, max_len)
    
    # 解码器部分
    self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
    self.tgt_pos = PositionalEncoding(d_model, max_len)
    
    # 编码器和解码器堆叠
    encoder_layer = EncoderLayer(d_model, nhead, d_ff, dropout)
    decoder_layer = DecoderLayer(d_model, nhead, d_ff, dropout)
    self.encoder = Encoder(encoder_layer, nlayers)
    self.decoder = Decoder(decoder_layer, nlayers)
    
    # 输出层
    self.generator = nn.Linear(d_model, tgt_vocab)
    
    # 权重初始化
    self._reset_parameters()
    
  def _reset_parameters(self):
    for p in self.parameters():
      if p.dim() > 1:
          nn.init.xavier_uniform_(p)
              
  def forward(self, src, tgt, src_mask, tgt_mask):
    # 编码源序列
    memory = self.encode(src, src_mask)
    
    # 解码目标序列
    output = self.decode(memory, src_mask, tgt, tgt_mask)
    
    # 生成预测
    return self.generator(output)
  
  def encode(self, src, src_mask):
    src_embedded = self.src_pos(self.src_embed(src))
    return self.encoder(src_embedded, src_mask)
  
  def decode(self, memory, src_mask, tgt, tgt_mask):
    tgt_embedded = self.tgt_pos(self.tgt_embed(tgt))
    return self.decoder(tgt_embedded, memory, src_mask, tgt_mask)

# 创建掩码的辅助函数
def create_mask(src, tgt, pad_idx):
  src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
  
  if tgt is not None:
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
    seq_len = tgt.size(1)
    nopeak_mask = torch.tril(torch.ones(seq_len, seq_len)).type_as(tgt_mask)
    tgt_mask = tgt_mask & nopeak_mask
  else:
    tgt_mask = None
      
  return src_mask, tgt_mask