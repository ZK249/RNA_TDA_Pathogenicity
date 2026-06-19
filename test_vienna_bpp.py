#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViennaRNA BPP 提取测试
"""

import numpy as np

try:
    import RNA
except ImportError:
    print("[ERROR] 未安装 ViennaRNA。请运行: conda install bioconda::viennarna")
    exit(1)


def test_bpp(sequence: str):
    print(f"序列: {sequence}")
    print(f"长度: {len(sequence)}")
    print()

    # 1. MFE
    fc = RNA.fold_compound(sequence)
    mfe_struct, mfe_energy = fc.mfe()
    print(f"MFE 结构: {mfe_struct}")
    print(f"MFE 能量: {mfe_energy:.2f} kcal/mol")
    print()

    # 2. BPP
    fc = RNA.fold_compound(sequence)
    fc.pf()
    bpp = fc.bpp()

    print(f"BPP 类型: {type(bpp)}")
    print()

    # 3. 提取为 numpy 矩阵
    n = len(sequence)
    mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i < j:
                try:
                    mat[i][j] = bpp[i + 1][j + 1]
                    mat[j][i] = mat[i][j]
                except Exception:
                    pass

    # 4. 打印矩阵左上角（5x5）
    print("BPP 矩阵 (前 5x5):")
    print("     " + "  ".join(f"{sequence[i]:>3}" for i in range(min(n, 5))))
    for i in range(min(n, 5)):
        row = "  ".join(f"{mat[i][j]:.2f}" for j in range(min(n, 5)))
        print(f"{sequence[i]:>3}  {row}")
    print()

    # 5. 打印强配对 (>0.5)
    print("强配对概率 (>0.5):")
    found = False
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i][j] > 0.5:
                print(f"  {i + 1}({sequence[i]}) - {j + 1}({sequence[j]}): {mat[i][j]:.3f}")
                found = True
    if not found:
        print("  无")
    print()

    # 6. 距离矩阵（翻转）
    dist = np.ones((n, n))
    for i in range(n - 1):
        dist[i][i + 1] = 0.1
        dist[i + 1][i] = 0.1
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i][j] > 0.01:
                dist[i][j] = 1.0 - mat[i][j]
                dist[j][i] = dist[i][j]

    print("距离矩阵 (前 5x5):")
    print("     " + "  ".join(f"{sequence[i]:>3}" for i in range(min(n, 5))))
    for i in range(min(n, 5)):
        row = "  ".join(f"{dist[i][j]:.2f}" for j in range(min(n, 5)))
        print(f"{sequence[i]:>3}  {row}")

    return mat, dist, mfe_struct, mfe_energy


if __name__ == "__main__":
    # 测试序列：一个简单的茎环结构
    seq = "GGGAAACCC"
    test_bpp(seq)
