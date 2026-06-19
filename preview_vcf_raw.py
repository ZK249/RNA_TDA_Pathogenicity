#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClinVar VCF 原始数据预览
"""

import argparse
import gzip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", default="./data/clinvar.vcf.gz")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    with gzip.open(args.vcf, "rt", encoding="utf-8") as f:
        # 打印头部（## 和 # 开头的行）
        header_lines = []
        for line in f:
            if line.startswith("##"):
                header_lines.append(line.strip())
            elif line.startswith("#"):
                header_lines.append(line.strip())
                break

        print("=== VCF 头部 ===")
        for h in header_lines:
            print(h)

        print(f"\n=== 前 {args.n} 条数据 ===\n")

        # 打印数据
        for i, line in enumerate(f, 1):
            if i > args.n:
                break
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue

            chrom, pos, id_, ref, alt, qual, filter_, info = parts[:8]

            # 解析 INFO 为 key=value 列表
            info_items = []
            for item in info.split(";"):
                if "=" in item:
                    k, v = item.split("=", 1)
                    # 截断超长值
                    if len(v) > 100:
                        v = v[:97] + "..."
                    info_items.append(f"{k}={v}")
                else:
                    info_items.append(item)

            info_str = " | ".join(info_items)

            print(f"[{i}] {chrom}\t{pos}\t{id_}\t{ref}>{alt}\t{qual}\t{filter_}")
            print(f"    INFO: {info_str}")
            print()


if __name__ == "__main__":
    main()
