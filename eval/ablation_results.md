# APEX Component Ablation Matrix Results

> Generated: 2026-10-27 17:10:54 | Model: `gemma3:1b`

| Variant | Token F1 | 95% CI | ROUGE-L | Faithfulness | Context Toks | Latency (s) | Rejection Acc | vs. Full (p) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full APEX** | 41.8 | [29.6, 56.7] | 41.0 | 100.0% | 286 | 3.76 | 100.0% | - |
| **w/o Silhouette Gate** | 41.8 | [29.6, 56.7] | 41.0 | 100.0% | 286 | 4.12 | 100.0% | p=1.000 |
| **w/o Dynamic BIC** | 41.1 | [29.3, 54.7] | 40.0 | 100.0% | 286 | 4.26 | 100.0% | p=0.199 |
| **w/o Soft Clustering** | 41.8 | [29.6, 56.7] | 41.0 | 100.0% | 286 | 4.10 | 100.0% | p=1.000 |
| **w/o BM25 RRF** | 41.6 | [29.4, 56.7] | 40.7 | 100.0% | 286 | 4.18 | 100.0% | p=0.665 |
| **w/o Jaccard Dedup** | 41.5 | [29.7, 55.2] | 40.8 | 100.0% | 286 | 4.31 | 100.0% | p=0.753 |
| **Baseline: Naive RAG** | 40.9 | [27.7, 55.1] | 40.2 | 95.0% | 286 | 4.33 | 100.0% | p=0.838 |

---

## LaTeX Source Code (`tab:ablation_matrix`)

```latex
\begin{table*}[t]
\centering
\small
\begin{tabular}{l c c c c c c c}
\toprule
\textbf{Ablation Variant} & \textbf{Token F1} & \textbf{95\% CI} & \textbf{ROUGE-L} & \textbf{Faithfulness} & \textbf{Tokens} & \textbf{Latency (s)} & \textbf{$p$-value} \\
\midrule
\textbf{Full APEX (Ours)} & \textbf{41.8} & [29.6, 56.7] & 41.0 & 100.0\% & 286 & 3.76 & --- \\
w/o Silhouette Gate & 41.8 & [29.6, 56.7] & 41.0 & 100.0\% & 286 & 4.12 & $1.000$ \\
w/o Dynamic BIC & 41.1 & [29.3, 54.7] & 40.0 & 100.0\% & 286 & 4.26 & $0.199$ \\
w/o Soft Clustering & 41.8 & [29.6, 56.7] & 41.0 & 100.0\% & 286 & 4.10 & $1.000$ \\
w/o BM25 RRF & 41.6 & [29.4, 56.7] & 40.7 & 100.0\% & 286 & 4.18 & $0.665$ \\
w/o Jaccard Dedup & 41.5 & [29.7, 55.2] & 40.8 & 100.0\% & 286 & 4.31 & $0.753$ \\
Baseline: Naive RAG & 40.9 & [27.7, 55.1] & 40.2 & 95.0\% & 286 & 4.33 & $0.838$ \\
\bottomrule
\end{tabular}
\caption{Component ablation matrix evaluating the impact of individual architectural choices in APEX. $^{**}p < 0.01$, $^{*}p < 0.05$ against Full APEX via paired $t$-test.}
\label{tab:ablation_matrix}
\end{table*}
```
