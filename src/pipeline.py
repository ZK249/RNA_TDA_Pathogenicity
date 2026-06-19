#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNA-TDA Pathogenicity Pipeline
端到端预测：RNA 序列 -> 结构建模 -> 拓扑特征 -> 轻量 Transformer + RAG -> 致病性概率

使用示例：
    from src.pipeline import RNAPathogenicityPipeline

    pipeline = RNAPathogenicityPipeline(db_path="./chroma_db")

    # 预测一个变异
    result = pipeline.predict(
        wild_seq="GGGAAACCC",
        variant_pos=4,
        variant_nt="U"
    )

    print(result["pathogenicity_prob"])
"""

import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple

import RNA
from ripser import ripser

from src.rag.knowledge_base import BirthDefectRAG


# ==========================================
# 1. 结构建模层
# ==========================================
class RNAStructureModeler:
    """ViennaRNA 封装：BPP 矩阵 + MFE 结构。"""

    def __init__(self, temperature: float = 37.0):
        self.md = RNA.md()
        self.md.temperature = temperature

    def get_bpp_matrix(self, sequence: str) -> np.ndarray:
        """返回碱基配对概率矩阵 (n x n)。"""
        n = len(sequence)
        mat = np.zeros((n, n))

        fc = RNA.fold_compound(sequence, self.md)
        fc.pf()
        bpp = fc.bpp()

        for i in range(n):
            for j in range(n):
                if i < j:
                    try:
                        mat[i][j] = bpp[i + 1][j + 1]
                        mat[j][i] = mat[i][j]
                    except Exception:
                        pass
        return mat

    def get_mfe(self, sequence: str) -> Tuple[str, float]:
        """返回 MFE 结构和自由能。"""
        fc = RNA.fold_compound(sequence, self.md)
        mfe_struct, mfe_energy = fc.mfe()
        return mfe_struct, mfe_energy


# ==========================================
# 2. 拓扑特征层
# ==========================================
class RNATopologicalFingerprint:
    """Ripser 持久同调：提取结构稳定性指纹。"""

    def __init__(self, max_dim: int = 1):
        self.max_dim = max_dim

    def bpp_to_distance(self, bpp: np.ndarray) -> np.ndarray:
        """BPP -> 距离矩阵。"""
        n = len(bpp)
        dist = np.ones((n, n))

        # 骨架相邻
        for i in range(n - 1):
            dist[i, i + 1] = 0.1
            dist[i + 1, i] = 0.1

        # 碱基配对
        for i in range(n):
            for j in range(i + 1, n):
                if bpp[i, j] > 0.01:
                    dist[i, j] = 1.0 - bpp[i, j]
                    dist[j, i] = dist[i, j]
        return dist

    def extract(self, bpp: np.ndarray) -> Dict:
        """提取拓扑特征向量。"""
        dist = self.bpp_to_distance(bpp)
        result = ripser(dist, distance_matrix=True, maxdim=self.max_dim)
        dgms = result["dgms"]

        features = {}

        # 0 维
        dgm0 = dgms[0]
        if len(dgm0) > 0:
            finite0 = dgm0[dgm0[:, 1] != np.inf]
            features["betti0_count"] = len(finite0)
            features["mean_persist_0"] = float(np.mean(finite0[:, 1] - finite0[:, 0])) if len(finite0) > 0 else 0.0
            features["max_persist_0"] = float(np.max(finite0[:, 1] - finite0[:, 0])) if len(finite0) > 0 else 0.0

        # 1 维
        dgm1 = dgms[1]
        if len(dgm1) > 0:
            persists = dgm1[:, 1] - dgm1[:, 0]
            # 过滤噪声：只保留寿命 > 0.05 的显著环
            significant = persists > 0.05
            features["betti1_count"] = int(significant.sum())
            if significant.any():
                features["mean_persist_1"] = float(persists[significant].mean())
                features["max_persist_1"] = float(persists[significant].max())
                features["std_persist_1"] = float(persists[significant].std())
            else:
                features["mean_persist_1"] = 0.0
                features["max_persist_1"] = 0.0
                features["std_persist_1"] = 0.0
        else:
            features["betti1_count"] = 0
            features["mean_persist_1"] = 0.0
            features["max_persist_1"] = 0.0
            features["std_persist_1"] = 0.0

        # 固定长度特征向量（8 维）
        vec = np.array([
            features["betti0_count"],
            features["mean_persist_0"],
            features["max_persist_0"],
            features["betti1_count"],
            features["mean_persist_1"],
            features["max_persist_1"],
            features["std_persist_1"],
            len(bpp) / 100.0,  # 序列长度归一化
        ], dtype=np.float32)

        features["vector"] = vec
        return features


# ==========================================
# 3. K-mer Tokenizer
# ==========================================
class KmerTokenizer:
    """RNA K-mer 分词器。"""

    def __init__(self, k: int = 3, vocab_size: int = 256):
        self.k = k
        self.vocab_size = vocab_size
        # 4^k 种可能的 K-mer，但 vocab_size 限制为 256
        self.kmer_to_id = {}
        self._build_vocab()

    def _build_vocab(self):
        """构建 K-mer -> ID 映射。"""
        bases = ["A", "U", "G", "C"]
        from itertools import product

        all_kmers = ["".join(p) for p in product(bases, repeat=self.k)]
        # 只取前 vocab_size-1 个，保留 0 给 padding
        for idx, kmer in enumerate(all_kmers[: self.vocab_size - 1]):
            self.kmer_to_id[kmer] = idx + 1

    def encode(self, sequence: str) -> List[int]:
        """将序列编码为 token ID 列表。"""
        tokens = []
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i : i + self.k]
            tokens.append(self.kmer_to_id.get(kmer, 0))
        return tokens

    def pad(self, tokens: List[int], max_len: int = 128) -> np.ndarray:
        """填充/截断到固定长度。"""
        arr = np.zeros(max_len, dtype=np.int64)
        length = min(len(tokens), max_len)
        arr[:length] = tokens[:length]
        return arr


# ==========================================
# 4. 轻量 Transformer 分类器
# ==========================================
class LightweightPathogenicityTransformer(nn.Module):
    """
    ~200K 参数，融合 K-mer 序列编码 + 拓扑特征。
    """

    def __init__(
        self,
        vocab_size: int = 256,
        seq_embed_dim: int = 64,
        topo_dim: int = 8,
        num_heads: int = 4,
        num_layers: int = 2,
        d_model: int = 128,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_embed = nn.Embedding(vocab_size, seq_embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, seq_embed_dim)
        self.seq_proj = nn.Linear(seq_embed_dim, d_model)

        self.topo_proj = nn.Linear(topo_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model * 2,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),  # [良性, 致病]
        )

        self._count_parameters()

    def forward(self, seq_tokens: torch.Tensor, topo_features: torch.Tensor) -> torch.Tensor:
        """
        seq_tokens: (batch, seq_len)
        topo_features: (batch, topo_dim)
        """
        batch_size, seq_len = seq_tokens.shape

        # 序列编码
        seq_emb = self.seq_embed(seq_tokens)  # (batch, seq_len, seq_embed_dim)
        positions = torch.arange(seq_len, device=seq_tokens.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(positions)
        seq_emb = seq_emb + pos_emb
        seq_emb = self.seq_proj(seq_emb)  # (batch, seq_len, d_model)

        # 拓扑特征
        topo_emb = self.topo_proj(topo_features).unsqueeze(1)  # (batch, 1, d_model)

        # 拼接
        combined = torch.cat([seq_emb, topo_emb.expand(-1, seq_len, -1)], dim=-1)  # (batch, seq_len, d_model*2)

        # Transformer
        out = self.transformer(combined)
        out = out.mean(dim=1)  # 全局平均池化

        return self.classifier(out)

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Parameters: {total / 1e3:.1f}K")


# ==========================================
# 5. 端到端 Pipeline
# ==========================================
class RNAPathogenicityPipeline:
    """
    端到端 ncRNA 致病性预测 Pipeline。
    """

    def __init__(
        self,
        db_path: str = "./chroma_db",
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.modeler = RNAStructureModeler()
        self.tda = RNATopologicalFingerprint()
        self.tokenizer = KmerTokenizer(k=3, vocab_size=256)
        self.rag = BirthDefectRAG(db_path=db_path)

        self.model = LightweightPathogenicityTransformer(
            vocab_size=256,
            seq_embed_dim=64,
            topo_dim=8,
            num_heads=4,
            num_layers=2,
            d_model=128,
        ).to(self.device)

        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[Pipeline] 加载模型: {model_path}")
        else:
            print("[Pipeline] 使用随机初始化模型（需训练）")

        self.model.eval()

    def _build_mutant(self, wild_seq: str, variant_pos: int, variant_nt: str) -> str:
        """构建突变序列。"""
        if variant_pos is None or variant_nt is None:
            return wild_seq
        return wild_seq[:variant_pos] + variant_nt + wild_seq[variant_pos + 1 :]

    def _get_structure(self, sequence: str) -> Dict:
        """结构建模：BPP + MFE。"""
        bpp = self.modeler.get_bpp_matrix(sequence)
        mfe_struct, mfe_energy = self.modeler.get_mfe(sequence)
        return {
            "bpp": bpp,
            "mfe_structure": mfe_struct,
            "mfe_energy": float(mfe_energy),
        }

    def _get_topo_features(self, bpp: np.ndarray) -> Dict:
        """拓扑特征提取。"""
        return self.tda.extract(bpp)

    def _get_sequence_tokens(self, sequence: str) -> torch.Tensor:
        """序列编码。"""
        tokens = self.tokenizer.encode(sequence)
        padded = self.tokenizer.pad(tokens, max_len=128)
        return torch.from_numpy(padded).unsqueeze(0).to(self.device)

    def _get_rag_evidence(self, sequence: str, variant_pos: Optional[int], variant_nt: Optional[str]) -> Dict:
        """RAG 检索证据。"""
        query = f"RNA sequence {sequence[:50]}"
        if variant_pos is not None:
            query += f" variant at position {variant_pos}"

        try:
            results = self.rag.retrieve(query, n_results=3)
            return results
        except Exception as e:
            print(f"[WARN] RAG 检索失败: {e}")
            return {"documents": [[]], "metadatas": [[]]}

    def predict(
        self,
        wild_seq: str,
        variant_pos: Optional[int] = None,
        variant_nt: Optional[str] = None,
    ) -> Dict:
        """
        端到端预测。

        Args:
            wild_seq: 野生型 RNA 序列
            variant_pos: 变异位置（0-based），None 表示无变异
            variant_nt: 突变碱基（A/U/G/C），None 表示无变异

        Returns:
            dict: 包含致病性概率、结构指标、证据链
        """
        # 1. 构建突变序列
        mut_seq = self._build_mutant(wild_seq, variant_pos, variant_nt)

        # 2. 结构建模
        wild_struct = self._get_structure(wild_seq)
        mut_struct = self._get_structure(mut_seq)

        # 3. 拓扑特征
        wild_topo = self._get_topo_features(wild_struct["bpp"])
        mut_topo = self._get_topo_features(mut_struct["bpp"])

        # 4. 序列编码
        seq_tokens = self._get_sequence_tokens(mut_seq)

        # 5. 拓扑特征张量
        topo_tensor = torch.from_numpy(mut_topo["vector"]).unsqueeze(0).to(self.device)

        # 6. RAG 检索
        evidence = self._get_rag_evidence(mut_seq, variant_pos, variant_nt)

        # 7. 模型预测（如果未训练，概率是随机的）
        with torch.no_grad():
            logits = self.model(seq_tokens, topo_tensor)
            probs = torch.softmax(logits, dim=-1)
            pathogenic_prob = float(probs[0][1].cpu().numpy())

        # 8. 结构变化指标
        struct_change = {
            "mfe_energy_delta": mut_struct["mfe_energy"] - wild_struct["mfe_energy"],
            "betti1_count_delta": mut_topo["betti1_count"] - wild_topo["betti1_count"],
            "mean_persist_1_delta": mut_topo["mean_persist_1"] - wild_topo["mean_persist_1"],
            "max_persist_1_delta": mut_topo["max_persist_1"] - wild_topo["max_persist_1"],
        }

        return {
            "pathogenicity_prob": pathogenic_prob,
            "is_pathogenic": pathogenic_prob > 0.5,
            "wild_type": {
                "sequence": wild_seq,
                "mfe_structure": wild_struct["mfe_structure"],
                "mfe_energy": wild_struct["mfe_energy"],
                "topo_features": {k: v for k, v in wild_topo.items() if k != "vector"},
            },
            "mutant": {
                "sequence": mut_seq,
                "mfe_structure": mut_struct["mfe_structure"],
                "mfe_energy": mut_struct["mfe_energy"],
                "topo_features": {k: v for k, v in mut_topo.items() if k != "vector"},
            },
            "structural_change": struct_change,
            "evidence": evidence,
        }

    def train_step(self, seq_tokens, topo_features, labels, optimizer, criterion):
        """单步训练（用于后续训练）。"""
        self.model.train()
        optimizer.zero_grad()
        logits = self.model(seq_tokens, topo_features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        return float(loss.item())


# ==========================================
# 6. 命令行测试
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RNA-TDA 致病性预测 Pipeline")
    parser.add_argument("--seq", default="GGGAAACCC", help="野生型 RNA 序列")
    parser.add_argument("--pos", type=int, default=None, help="变异位置（0-based）")
    parser.add_argument("--nt", default=None, help="突变碱基（A/U/G/C）")
    parser.add_argument("--db-path", default="./chroma_db", help="ChromaDB 路径")
    parser.add_argument("--model-path", default=None, help="预训练模型路径")
    args = parser.parse_args()

    print("=" * 60)
    print("RNA-TDA Pathogenicity Pipeline")
    print("=" * 60)

    pipeline = RNAPathogenicityPipeline(
        db_path=args.db_path,
        model_path=args.model_path,
    )

    result = pipeline.predict(
        wild_seq=args.seq,
        variant_pos=args.pos,
        variant_nt=args.nt,
    )

    print(f"\n野生型序列: {result['wild_type']['sequence']}")
    print(f"突变序列:   {result['mutant']['sequence']}")
    print(f"MFE 结构:   {result['mutant']['mfe_structure']}")
    print(f"MFE 能量:   {result['mutant']['mfe_energy']:.2f}")
    print()

    print("拓扑特征（突变型）:")
    for k, v in result["mutant"]["topo_features"].items():
        print(f"  {k}: {v}")
    print()

    print("结构变化:")
    for k, v in result["structural_change"].items():
        print(f"  {k}: {v:.4f}")
    print()

    print(f"致病性概率: {result['pathogenicity_prob']:.4f}")
    print(f"预测结果:   {'致病' if result['is_pathogenic'] else '良性'}")
    print()

    if result["evidence"] and result["evidence"].get("documents"):
        docs = result["evidence"]["documents"][0]
        if docs:
            print(f"RAG 证据（Top {len(docs)}）:")
            for i, doc in enumerate(docs[:3], 1):
                print(f"  [{i}] {doc[:120]}...")