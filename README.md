# RNA-TDA Pathogenicity: ncRNA 变异与出生缺陷辅助分析系统

> **目标**：针对出生缺陷三级防控中 **ncRNA 变异解读的空白**，构建基于拓扑数据分析（TDA）的结构稳定性量化系统，为印记区 ncRNA（H19、MEG3、KCNQ1OT1、SNORD116 等）的致病变异提供辅助分析工具。

---

## 背景与问题

现有变异注释工具（SIFT、PolyPhen、CADD）全部是训练在**蛋白质编码区**上的，核心假设是"变异改变氨基酸 → 改变蛋白质功能"。但 ncRNA **没有氨基酸**，其功能高度依赖**二级结构（茎环）**，现有工具对 ncRNA 完全失效。

**本项目尝试的新思路**：不看病码子，直接量化 ncRNA 结构稳定性。结构破坏大的变异 → 功能丧失 → 出生缺陷风险上升。

---

## 核心方法

```
ncRNA 序列 + 变异位点
    ↓
ViennaRNA → 碱基配对概率矩阵 (BPP)
    ↓
Ripser 持久同调 → 拓扑指纹（茎环寿命、结构域连通性）
    ↓
轻量 Transformer (~200K 参数) → 致病性概率
    ↓
ClinVar RAG → 检索同类 ncRNA 致病变异证据
```

| 组件 | 作用 | 对应生物学问题 |
|------|------|---------------|
| **ViennaRNA** | 预测 RNA 二级结构，提取碱基配对概率 | ncRNA 的功能基础是茎环，结构必须先算出来 |
| **Ripser (TDA)** | 持久同调量化茎环稳定性 | 茎环寿命长的 = 结构结实；寿命短的 = 容易崩 |
| **轻量 Transformer** | 融合 K-mer 序列语义 + 拓扑特征 | 综合判断结构破坏程度与致病概率 |
| **ChromaDB + ClinVar** | 向量知识库检索历史致病变异 | 提供可解释的证据链（不是 AI 瞎猜） |

---

## 目标基因

聚焦 **出生缺陷相关印记区 ncRNA**，基于 ClinVar/OMIM 的明确致病记录筛选：

| 基因 | 类型 | 关联出生缺陷 |
|------|------|-------------|
| **H19** | lncRNA | Beckwith-Wiedemann 综合征、Silver-Russell 综合征 |
| **MEG3** | lncRNA | 胎儿生长受限、先兆子痫 |
| **KCNQ1OT1** | lncRNA | BWS、长 QT 综合征 |
| **SNORD116** | snoRNA | **Prader-Willi 综合征**（核心致病分子） |
| **SNORD115** | snoRNA | PWS / Angelman 综合征 |
| **XIST** | lncRNA | Turner 综合征、Klinefelter 综合征（性染色体失活） |
| **HAND2-AS1** | lncRNA | 心脏发育缺陷 |
| **FOXP4-AS1** | lncRNA | 房室间隔缺损 |

---

## 项目结构

```
RNA_TDA_PATHOGENICITY/
├── checkpoints/              # 模型权重
│   ├── best_model.pt
│   └── final_model.pt
├── chroma_db/               # ChromaDB 向量知识库（297 条 ClinVar ncRNA 记录）
├── data/
│   └── clinvar.vcf.gz       # ClinVar 全库 VCF（手动下载）
├── notebook/
│   └── demo.ipynb           # KCNQ1OT1 案例端到端演示
├── src/
│   ├── pipeline.py          # 核心 Pipeline
│   ├── train.py             # 模拟数据生成与训练
│   └── rag/
│       └── knowledge_base.py    # ClinVar 知识库构建
├── build_kb_strict.py       # ClinVar VCF 筛选脚本
├── preview_vcf_raw.py       # VCF 预览工具
├── test_vienna_bpp.py       # ViennaRNA 环境测试
├── test_ripser.py           # Ripser 环境测试
├── validate_kb.py           # 知识库验证
└── README.md
```

---

## 快速开始

### 1. 环境准备

```bash
conda create -n rna python=3.10
conda activate rna
pip install torch numpy scikit-learn pandas tqdm
pip install ripser persim chromadb sentence-transformers
conda install -c bioconda viennaRNA
```

### 2. 构建 ClinVar 知识库

```bash
# 下载 ClinVar VCF（~200MB）到 data/ 目录
# https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz

python build_kb_strict.py --vcf ./data/clinvar.vcf.gz --db-path ./chroma_db
```

**筛选结果**：从 ~900 万条记录中保留 **297 条 ncRNA Pathogenic/Likely_pathogenic** 记录，覆盖 H19、MEG3、KCNQ1OT1、SNORD116 等基因。

### 3. 训练模型（可选）

```bash
python src/train.py --n-samples 5000 --epochs 50 --batch-size 32 --lr 1e-4
```

### 4. 端到端预测

```bash
python src/pipeline.py \
    --seq "AUGCGAUCGCGGCUGCGCUGCGCUGCGCUGCGCUGCGCUGCGCUGCGCUGCGCUGCGCUGCGCUGCGCGGCG" \
    --pos 15 \
    --nt "A" \
    --model-path ./checkpoints/best_model.pt \
    --db-path ./chroma_db
```

### 5. Jupyter 演示

```bash
jupyter notebook notebook/demo.ipynb
```

---

## 数据说明

### 真实数据：ClinVar ncRNA 知识库
- **来源**：NCBI ClinVar VCF（月度更新）
- **筛选**：Pathogenic/Likely_pathogenic + 基因在白名单（H19/MEG3/KCNQ1OT1/SNORD116 等）
- **数量**：**297 条**
- **用途**：RAG 检索，提供真实证据链

### 模拟数据：结构扰动代理标签
- **生成方式**：基于随机茎环结构 + ViennaRNA 结构变化
- **标签逻辑**：MFE 能量上升 + 环寿命缩短 → 标记为致病
- **用途**：训练 Transformer 的概念验证
- **局限**：代理标签存在生物学简化，不代表真实功能实验

---

## 当前局限（诚实说明）

1. **真实 ncRNA 功能数据稀缺**：ClinVar 中明确标注为 Pathogenic 的 ncRNA 记录仅 ~300 条，且缺乏结构变化标签
2. **模拟标签粗糙**："结构破坏 = 致病"的假设过于简化，忽略了 RNA 冗余机制、别构效应、蛋白相互作用等复杂情况
3. **模型性能天花板**：模拟数据上 AUC 约 0.6，说明代理标签的区分度有限
4. **不是临床诊断工具**：当前为**概念验证系统**，验证 TDA + 轻量 AI + RAG 的技术架构可行性

---

## 为什么仍然有价值？

- **填补工具空白**：现有工具（SIFT/PolyPhen/CADD）对 ncRNA 完全失效，本项目提供了**结构稳定性量化**的新维度
- **技术栈差异化**：TDA + 轻量 Transformer + RAG 的组合在生物信息学领域有独特性
- **架构可扩展**：如果接入真实结构探测数据（SHAPE-MaP、DMS-MaPseq），Pipeline 可直接复用
- **知识库真实**：297 条 ClinVar ncRNA 记录是真实数据，RAG 检索的证据链有临床参考价值

---
## License

MIT
