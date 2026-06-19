#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ripser 持久同调测试（修复版）
"""

import numpy as np
from ripser import ripser


def test_ripser_on_rna(dist_matrix: np.ndarray):
    n = len(dist_matrix)
    print(f"距离矩阵大小: {n}x{n}")
    print()

    # 1. 计算持久同调
    result = ripser(dist_matrix, distance_matrix=True, maxdim=1)
    dgms = result['dgms']

    # 2. 0 维持久图（连通分量）
    dgm0 = dgms[0]
    print(f"0 维持久图: {len(dgm0)} 个点")
    for i, (birth, death) in enumerate(dgm0):
        if death == np.inf:
            print(f"  [{i}] birth={birth:.3f}, death=inf (永远不死)")
        else:
            persist = death - birth
            print(f"  [{i}] birth={birth:.3f}, death={death:.3f}, persist={persist:.3f}")
    print()

    # 3. 1 维持久图（环）
    dgm1 = dgms[1]
    print(f"1 维持久图: {len(dgm1)} 个点")
    if len(dgm1) == 0:
        print("  无环结构")
    else:
        for i, (birth, death) in enumerate(dgm1):
            persist = death - birth
            print(f"  [{i}] birth={birth:.3f}, death={death:.3f}, persist={persist:.3f}")
    print()

    # 4. 简单统计
    if len(dgm0) > 0:
        finite0 = dgm0[dgm0[:, 1] != np.inf]
        if len(finite0) > 0:
            mean_persist_0 = np.mean(finite0[:, 1] - finite0[:, 0])
            print(f"0 维平均寿命: {mean_persist_0:.4f}")
    if len(dgm1) > 0:
        mean_persist_1 = np.mean(dgm1[:, 1] - dgm1[:, 0])
        max_persist_1 = np.max(dgm1[:, 1] - dgm1[:, 0])
        print(f"1 维平均寿命: {mean_persist_1:.4f}")
        print(f"1 维最大寿命: {max_persist_1:.4f}")

    return dgms


if __name__ == "__main__":
    # 用 GGGAAACCC 的距离矩阵测试
    n = 9
    dist = np.ones((n, n))

    # 骨架相邻
    for i in range(n - 1):
        dist[i][i + 1] = 0.1
        dist[i + 1][i] = 0.1

    # 碱基配对（概率 ~0.52，距离 ~0.48）
    pairs = [(0, 8), (1, 7), (2, 6)]
    for i, j in pairs:
        dist[i][j] = 0.48
        dist[j][i] = 0.48

    print("=" * 50)
    print("Ripser 持久同调测试")
    print("=" * 50)
    print()
    test_ripser_on_rna(dist)