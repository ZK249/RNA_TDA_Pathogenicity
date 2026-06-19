#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClinVar VCF 严格筛选 + 知识库重建
只保留：
  1. 基因名明确在 ncRNA 白名单里
  2. CLNSIG 包含 Pathogenic/Likely_pathogenic（不含 Conflicting）
  3. 可选：疾病名包含出生缺陷关键词（更严格）

使用：
    python build_kb_strict.py --vcf ./data/clinvar.vcf.gz --db-path ./chroma_db
"""

import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import gzip
import shutil
from typing import List, Dict, Optional
from collections import Counter

from tqdm import tqdm
from src.rag.knowledge_base import BirthDefectRAG


# ==========================================
# 严格白名单：只认这些基因
# ==========================================
NCRNA_GENES = [
    "H19", "MEG3", "MEG8", "MEG9", "AIRN", "KCNQ1OT1",
    "MIR296", "MIRLET7BHG", "MIR17HG",
    "XIST", "NEAT1", "MALAT1", "SNHG14",
    "SNORD116", "SNORD115", "SNORD109",
    "RMST", "FOXP4-AS1", "LINC00261", "HAND2-AS1",
]

# 疾病关键词（可选的第三层过滤）
DISEASE_KEYWORDS = [
    "syndrome", "development", "growth", "fetal", "preeclampsia",
    "beckwith", "silver-russell", "prader-willi", "angelman",
    "birth defect", "congenital", "abnormality", "imprinting",
    "placental", "pregnancy", "neonatal", "infantile",
    "neonatal", "perinatal", "microcephaly", "macrocephaly",
    "overgrowth", "short stature", "skeletal dysplasia",
]


def parse_info(info_str: str) -> Dict[str, str]:
    result = {}
    if not info_str or info_str == ".":
        return result
    for item in info_str.split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            result[k] = v
        else:
            result[item] = "True"
    return result


def extract_genes(geneinfo: str) -> List[str]:
    if not geneinfo:
        return []
    return [g.split(":")[0] for g in geneinfo.split("|")]


def is_strict_pathogenic(clnsig: str) -> bool:
    """严格判断：必须是 Pathogenic 或 Likely_pathogenic，不含 Conflicting。"""
    if not clnsig:
        return False
    c = clnsig.lower()
    # 必须包含 pathogenic
    if "pathogenic" not in c:
        return False
    # 必须不含 conflicting
    if "conflicting" in c:
        return False
    return True


def is_whitelist_gene(geneinfo: str) -> bool:
    """基因名必须明确在白名单里。"""
    genes = extract_genes(geneinfo)
    return any(g in NCRNA_GENES for g in genes)


def has_disease_keyword(disease: str) -> bool:
    """疾病名是否包含出生缺陷关键词。"""
    if not disease or disease == ".":
        return False
    d = disease.lower().replace("_", " ")
    return any(k in d for k in DISEASE_KEYWORDS)


def build_knowledge_base(vcf_path: str, db_path: str, strict_disease: bool = False):
    print("=" * 60)
    print("ClinVar VCF 严格筛选 + 知识库重建")
    print("=" * 60)
    print(f"VCF: {vcf_path}")
    print(f"白名单基因: {len(NCRNA_GENES)} 个")
    print(f"严格疾病过滤: {'开启' if strict_disease else '关闭'}")
    print()

    # 删除旧知识库（彻底重建）
    db_path_obj = Path(db_path)
    if db_path_obj.exists():
        print(f"删除旧知识库: {db_path}")
        shutil.rmtree(db_path_obj)
        print("旧知识库已清除\n")

    rag = BirthDefectRAG(db_path=db_path)
    records = []

    stats = {
        "total": 0,
        "pathogenic": 0,
        "whitelist": 0,
        "disease_match": 0,
        "final": 0,
    }

    rejected_reasons = Counter()

    with gzip.open(vcf_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="扫描 VCF"):
            if line.startswith("#"):
                continue

            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue

            stats["total"] += 1
            chrom, pos, id_, ref, alt, qual, filter_, info = parts[:8]
            info_dict = parse_info(info)

            clnsig = info_dict.get("CLNSIG", "")
            geneinfo = info_dict.get("GENEINFO", "")
            disease = info_dict.get("CLNDN", "")
            mc = info_dict.get("MC", "")
            hgvs = info_dict.get("CLNHGVS", "")
            review = info_dict.get("CLNREVSTAT", "")
            vtype = info_dict.get("CLNVC", "")

            # 第一层：必须是 Pathogenic/Likely_pathogenic
            if not is_strict_pathogenic(clnsig):
                rejected_reasons["not_pathogenic"] += 1
                continue
            stats["pathogenic"] += 1

            # 第二层：基因必须在白名单里
            if not is_whitelist_gene(geneinfo):
                rejected_reasons["not_whitelist_gene"] += 1
                continue
            stats["whitelist"] += 1

            # 第三层（可选）：疾病关键词
            if strict_disease and not has_disease_keyword(disease):
                rejected_reasons["no_disease_keyword"] += 1
                continue
            if strict_disease:
                stats["disease_match"] += 1

            # 通过所有筛选，构建记录
            genes = extract_genes(geneinfo)
            primary_gene = next((g for g in genes if g in NCRNA_GENES), genes[0] if genes else "")

            text_parts = [f"ClinVar variation {id_}"]
            if primary_gene:
                text_parts.append(f"Gene: {primary_gene}")
            if hgvs:
                text_parts.append(f"Variant: {hgvs}")
            text_parts.append(f"Clinical significance: {clnsig}")
            if disease and disease != ".":
                text_parts.append(f"Disease: {disease}")
            if mc:
                text_parts.append(f"Molecular consequence: {mc}")
            if vtype:
                text_parts.append(f"Type: {vtype}")

            records.append({
                "id": f"clinvar_{id_}",
                "text": "; ".join(text_parts),
                "metadata": {
                    "gene": primary_gene,
                    "variation_id": id_,
                    "hgvs": hgvs,
                    "clinical_significance": clnsig,
                    "disease": disease,
                    "molecular_consequence": mc,
                    "variant_type": vtype,
                    "review_status": review,
                    "source": "ClinVar VCF",
                }
            })
            stats["final"] += 1

    # 打印统计
    print(f"\n{'='*60}")
    print("【筛选统计】")
    print(f"{'='*60}")
    print(f"总记录数:              {stats['total']:,}")
    print(f"Pathogenic 过滤后:     {stats['pathogenic']:,}")
    print(f"白名单基因过滤后:      {stats['whitelist']:,}")
    if strict_disease:
        print(f"疾病关键词过滤后:      {stats['disease_match']:,}")
    print(f"最终保留:              {stats['final']:,}")
    print()
    print("【被拒绝原因】")
    for reason, count in rejected_reasons.most_common():
        print(f"  {reason:30s} : {count:8,}")
    print()

    # 去重
    seen = set()
    unique_records = []
    for r in records:
        vid = r["metadata"]["variation_id"]
        if vid not in seen:
            seen.add(vid)
            unique_records.append(r)

    print(f"去重后: {len(unique_records)} 条")

    # 灌库
    if unique_records:
        batch_size = 50
        for i in range(0, len(unique_records), batch_size):
            batch = unique_records[i:i + batch_size]
            rag.add_records(batch)
            print(f"  已写入 {i+len(batch)}/{len(unique_records)}")
    else:
        print("警告：没有记录通过筛选，知识库为空！")

    print(f"\n知识库构建完成: {db_path}")
    return rag


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", default="./data/clinvar.vcf.gz", help="ClinVar VCF 路径")
    parser.add_argument("--db-path", default="./chroma_db", help="ChromaDB 存储路径")
    parser.add_argument("--strict-disease", action="store_true", 
                        help="开启严格疾病关键词过滤（可能过滤掉更多记录）")
    args = parser.parse_args()

    build_knowledge_base(args.vcf, args.db_path, args.strict_disease)
