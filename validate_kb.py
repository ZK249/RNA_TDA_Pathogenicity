#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChromaDB 知识库验证脚本
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag.knowledge_base import BirthDefectRAG

rag = BirthDefectRAG(db_path="./chroma_db")

# 1. 总记录数（不会爆 SQL）
total = rag.collection.count()
print(f"总记录数: {total}")

# 2. 看前 10 条（peek 不会爆）
print(f"\n{'='*60}")
print("【前 10 条记录预览】")
print(f"{'='*60}")

peek = rag.collection.peek(limit=10)
for i in range(len(peek['ids'])):
    print(f"\n--- 记录 {i+1} ---")
    print(f"ID: {peek['ids'][i]}")
    print(f"文本: {peek['documents'][i][:200]}...")
    meta = peek['metadatas'][i]
    print(f"基因: {meta.get('gene', 'N/A')}")
    print(f"疾病: {meta.get('disease', 'N/A')}")
    print(f"临床意义: {meta.get('clinical_significance', 'N/A')}")

# 3. 统计基因分布（分页查询）
print(f"\n{'='*60}")
print("【基因分布统计】")
print(f"{'='*60}")

from collections import Counter
gene_counter = Counter()
disease_counter = Counter()
sig_counter = Counter()

batch_size = 500
for offset in range(0, min(total, 5000), batch_size):  # 最多统计前5000条
    batch = rag.collection.get(limit=batch_size, offset=offset)
    for meta in batch['metadatas']:
        gene = meta.get('gene', '')
        disease = meta.get('disease', '')
        sig = meta.get('clinical_significance', '')
        if gene:
            gene_counter[gene] += 1
        if disease and disease not in ['.', 'not_provided', 'not_specified', '']:
            disease_counter[disease] += 1
        if sig:
            sig_counter[sig] += 1

print(f"\nTop 15 基因:")
for gene, count in gene_counter.most_common(15):
    print(f"  {gene:20s} : {count:4d}")

print(f"\nTop 10 疾病:")
for disease, count in disease_counter.most_common(10):
    print(f"  {disease:30s} : {count:3d}")

print(f"\n临床意义分布:")
for sig, count in sig_counter.most_common():
    print(f"  {sig:40s} : {count:4d}")