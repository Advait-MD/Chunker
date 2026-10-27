# 📄 APEX Conference Paper Manuscript

This directory contains the camera-ready LaTeX source code, BibTeX bibliography, and empirical evaluation tables for the APEX research paper targeting **ACL / EMNLP / ICLR / NeurIPS / MLSys**.

---

## 📑 Paper Title & Abstract

- **Title**: *APEX: Quality-Gated Recursive Semantic Clustering for Hallucination-Resistant Hierarchical Retrieval*
- **Authors**: Advait MD
- **Target Venues**: ACL (Information Retrieval / QA Track), EMNLP, ICLR, NeurIPS, MLSys.

---

## 🗂️ Directory Structure

```
paper/
├── main.tex             # Complete conference manuscript (8 Sections + Proofs)
├── references.bib       # Seminal references (Lewis et al., Sarthi et al., etc.)
└── README.md            # Compilation instructions & overview
```

---

## 🛠️ How to Compile to PDF

### Method 1: Using Standard LaTeX CLI (`pdflatex` + `bibtex`)

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Method 2: Using `latexmk` (Recommended)

```bash
cd paper
latexmk -pdf main.tex
```

### Method 3: Overleaf / arXiv Submission

1. Zip the `paper/` directory containing `main.tex` and `references.bib`.
2. Upload the `.zip` archive directly into [Overleaf](https://www.overleaf.com) or arXiv.

---

## 📊 Key Tables Included in the Paper

1. **Table 1 (`tab:main_benchmark`)**: Primary head-to-head empirical comparison across **NarrativeQA**, **QASPER**, and **QuALITY** benchmarks showing $+2.74$ to $+8.19$ Token F1 gains with $60\%$ fewer context tokens.
2. **Table 2 (`tab:multi_model_scaling`)**: Live cross-architecture scaling across **Google Gemma 3 1B** and **Meta Llama 3.2 1B**.
3. **Table 3 (`tab:ablation_matrix`)**: 7-row architectural component ablation matrix with 95% bootstrap confidence intervals and paired Student's $t$-test significance values.
4. **Table 4 (`tab:systems_efficiency`)**: Systems, compute, theoretical forward FLOPs ($2 \times N_{\text{params}} \times T_{\text{total}}$), and inference cost per 1M production queries.
