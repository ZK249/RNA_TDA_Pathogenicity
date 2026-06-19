import torch
import torch.nn as nn

class LightweightPathogenicityTransformer(nn.Module):
    """
    ~200K参数，复用GeneInsight的极简架构
    """
    def __init__(
        self,
        seq_vocab_size=256,      # K-mer tokenizer词汇表
        seq_embed_dim=64,
        topo_dim=40,           # 拓扑特征维度
        num_heads=4,
        num_layers=2,
        d_model=128,
        num_classes=2
    ):
        super().__init__()
        
        # 序列编码分支
        self.seq_embed = nn.Embedding(seq_vocab_size, seq_embed_dim)
        self.seq_proj = nn.Linear(seq_embed_dim, d_model)
        
        # 拓扑特征投影
        self.topo_proj = nn.Linear(topo_dim, d_model)
        
        # 极简Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model * 2,  # 拼接后维度
            nhead=num_heads,
            dim_feedforward=256,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        self._count_parameters()
    
    def forward(self, seq_tokens, topo_features):
        # seq_tokens: (batch, seq_len)
        # topo_features: (batch, topo_dim)
        
        seq_emb = self.seq_embed(seq_tokens)  # (batch, seq_len, seq_embed_dim)
        seq_emb = self.seq_proj(seq_emb)      # (batch, seq_len, d_model)
        
        topo_emb = self.topo_proj(topo_features).unsqueeze(1)  # (batch, 1, d_model)
        
        # 拼接序列和拓扑特征
        combined = torch.cat([seq_emb, topo_emb], dim=-1)  # (batch, seq_len, d_model*2)
        
        # Transformer处理
        out = self.transformer(combined)
        out = out.mean(dim=1)  # 全局平均池化
        
        return self.classifier(out)
    
    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model parameters: {total/1e3:.1f}K")