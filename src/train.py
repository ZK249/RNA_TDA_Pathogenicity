#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNA-TDA 模拟数据生成 + 模型训练脚本

使用：
    python src/train.py --n-samples 5000 --epochs 50 --batch-size 32 --lr 1e-4
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import json

from src.pipeline import (
    RNAStructureModeler,
    RNATopologicalFingerprint,
    KmerTokenizer,
    LightweightPathogenicityTransformer,
)


# ==========================================
# 1. 模拟数据生成
# ==========================================
COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}


def generate_stem_loop_sequence(length: int = 80, gc_content: float = 0.5) -> str:
    """生成茎环结构 RNA 序列。"""
    bases = ["A", "U", "G", "C"]

    five_prime_len = random.randint(10, 15)
    five_prime = "".join(random.choices(bases, k=five_prime_len))

    stem_len = random.randint(15, 25)
    stem_half = []
    for _ in range(stem_len):
        if random.random() < gc_content:
            stem_half.append(random.choice(["G", "C"]))
        else:
            stem_half.append(random.choice(["A", "U"]))
    stem_half = "".join(stem_half)
    stem_other = "".join(COMPLEMENT[b] for b in stem_half[::-1])

    loop_len = random.randint(5, 10)
    loop = "".join(random.choices(bases, k=loop_len))

    three_prime_len = random.randint(10, 15)
    three_prime = "".join(random.choices(bases, k=three_prime_len))

    sequence = five_prime + stem_half + loop + stem_other + three_prime

    while len(sequence) < length:
        sequence = random.choice(bases) + sequence + random.choice(bases)

    return sequence[:length]


def introduce_mutation(sequence: str) -> Tuple[str, int, str]:
    """引入随机点突变。"""
    seq_list = list(sequence)
    n = len(sequence)
    pos = random.randint(5, n - 6)
    original = seq_list[pos]
    choices = [b for b in "AUGC" if b != original]
    new_base = random.choice(choices)
    seq_list[pos] = new_base
    return "".join(seq_list), pos, new_base


# ==========================================
# 2. 数据集生成器
# ==========================================
class SyntheticRNADataset:
    def __init__(self, n_samples: int = 5000, seq_length: int = 80, random_seed: int = 42):
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.modeler = RNAStructureModeler()
        self.tda = RNATopologicalFingerprint()
        self.tokenizer = KmerTokenizer(k=3, vocab_size=256)

        random.seed(random_seed)
        np.random.seed(random_seed)

    def _compute_features(self, sequence: str) -> Dict:
        bpp = self.modeler.get_bpp_matrix(sequence)
        mfe_struct, mfe_energy = self.modeler.get_mfe(sequence)
        topo = self.tda.extract(bpp)

        tokens = self.tokenizer.encode(sequence)
        padded = self.tokenizer.pad(tokens, max_len=128)

        return {
            "seq_tokens": padded,
            "topo_vector": topo["vector"],
            "mfe_energy": float(mfe_energy),
            "betti1_count": topo["betti1_count"],
            "mean_persist_1": topo["mean_persist_1"],
            "max_persist_1": topo["max_persist_1"],
        }

    def generate(self) -> List[Dict]:
        data = []

        print(f"正在生成 {self.n_samples} 条模拟数据...")
        for _ in tqdm(range(self.n_samples), desc="Generating"):
            wild_seq = generate_stem_loop_sequence(self.seq_length)
            mut_seq, var_pos, var_nt = introduce_mutation(wild_seq)

            wild_features = self._compute_features(wild_seq)
            mut_features = self._compute_features(mut_seq)

            energy_delta = mut_features["mfe_energy"] - wild_features["mfe_energy"]
            persist_delta = mut_features["mean_persist_1"] - wild_features["mean_persist_1"]
            betti1_delta = mut_features["betti1_count"] - wild_features["betti1_count"]

            score = (
                energy_delta * 2.0
                + persist_delta * (-5.0)
                + betti1_delta * (-1.0)
            )

            label = 1 if score > 0 else 0

            data.append({
                "seq_tokens": torch.from_numpy(mut_features["seq_tokens"]).long(),
                "topo_vector": torch.from_numpy(mut_features["topo_vector"]).float(),
                "label": torch.tensor(label).long(),
                "wild_seq": wild_seq,
                "mut_seq": mut_seq,
                "var_pos": var_pos,
                "var_nt": var_nt,
                "score": score,
                "energy_delta": energy_delta,
                "persist_delta": persist_delta,
            })

        labels = [d["label"].item() for d in data]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        print(f"\n数据集统计: 总样本={len(data)}, 致病={n_pos}({n_pos/len(data)*100:.1f}%), 良性={n_neg}({n_neg/len(data)*100:.1f}%)")

        return data


# ==========================================
# 3. PyTorch Dataset
# ==========================================
class RNADataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "seq_tokens": item["seq_tokens"],
            "topo_vector": item["topo_vector"],
            "label": item["label"],
        }


# ==========================================
# 4. 训练
# ==========================================
def train_model(
    n_samples: int = 5000,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cpu",
    save_dir: str = "./checkpoints",
):
    device = torch.device(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    generator = SyntheticRNADataset(n_samples=n_samples)
    all_data = generator.generate()

    train_data, val_data = train_test_split(
        all_data, test_size=0.2, random_state=42,
        stratify=[d["label"].item() for d in all_data]
    )

    train_loader = DataLoader(RNADataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RNADataset(val_data), batch_size=batch_size, shuffle=False)

    model = LightweightPathogenicityTransformer(
        vocab_size=256, seq_embed_dim=64, topo_dim=8,
        num_heads=4, num_layers=2, d_model=128,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": []}

    print(f"\n{'='*60}")
    print("开始训练")
    print(f"{'='*60}")
    print(f"设备: {device} | 训练: {len(train_data)} | 验证: {len(val_data)} | batch: {batch_size} | lr: {lr} | epochs: {epochs}")

    for epoch in range(epochs):
        # 训练
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            seq_tokens = batch["seq_tokens"].to(device)
            topo_vector = batch["topo_vector"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(seq_tokens, topo_vector)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        avg_train_loss = np.mean(train_losses)

        # 验证
        model.eval()
        val_losses = []
        all_labels, all_probs, all_preds = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                seq_tokens = batch["seq_tokens"].to(device)
                topo_vector = batch["topo_vector"].to(device)
                labels = batch["label"].to(device)

                logits = model(seq_tokens, topo_vector)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())

                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
                all_preds.extend(preds)

        avg_val_loss = np.mean(val_losses)
        val_auc = roc_auc_score(all_labels, all_probs)
        val_acc = accuracy_score(all_labels, all_preds)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_auc"].append(val_auc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  ✓ 保存最佳模型 (AUC={val_auc:.4f})")

    torch.save(model.state_dict(), save_dir / "final_model.pt")
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print("训练完成")
    print(f"{'='*60}")
    print(f"最佳验证 AUC: {best_auc:.4f}")
    print(f"模型保存: {save_dir}/best_model.pt, {save_dir}/final_model.pt")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNA-TDA 模拟数据生成与训练")
    parser.add_argument("--n-samples", type=int, default=5000, help="模拟样本数")
    parser.add_argument("--seq-length", type=int, default=80, help="RNA 序列长度")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--device", default="cpu", help="设备 (cpu/cuda)")
    parser.add_argument("--save-dir", default="./checkpoints", help="模型保存路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_model(
        n_samples=args.n_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
    )