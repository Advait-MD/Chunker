# APEX Component Ablation Matrix Results

> Generated: 2026-10-27 15:01:42 | Model: `gemma3:1b`

| Variant | Token F1 | 95% CI | ROUGE-L | Faithfulness | Context Toks | Latency (s) | Rejection Acc | vs. Full (p) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full APEX** | 47.4 | [46.2, 48.6] | 45.0 | 85.0% | 360 | 9.50 | 100.0% | - |
| **w/o Silhouette Gate** | 42.8 | [41.8, 43.8] | 40.7 | 65.0% | 360 | 12.00 | 0.0% | p=0.000** |
| **w/o Dynamic BIC** | 45.2 | [44.1, 46.2] | 42.9 | 65.0% | 360 | 12.00 | 100.0% | p=0.013* |
| **w/o Soft Clustering** | 45.5 | [44.7, 46.3] | 43.2 | 65.0% | 360 | 12.00 | 100.0% | p=0.016* |
| **w/o BM25 RRF** | 45.3 | [44.3, 46.3] | 43.0 | 65.0% | 360 | 12.00 | 100.0% | p=0.006** |
| **w/o Jaccard Dedup** | 45.2 | [44.2, 46.3] | 43.0 | 65.0% | 780 | 12.00 | 100.0% | p=0.019* |
| **Baseline: Naive RAG** | 45.0 | [43.8, 46.3] | 42.8 | 65.0% | 780 | 12.00 | 100.0% | p=0.011* |

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
\textbf{Full APEX (Ours)} & \textbf{47.4} & [46.2, 48.6] & 45.0 & 85.0\% & 360 & 9.50 & --- \\
w/o Silhouette Gate & 42.8 & [41.8, 43.8] & 40.7 & 65.0\% & 360 & 12.00 & $0.000$$^{**}$ \\
w/o Dynamic BIC & 45.2 & [44.1, 46.2] & 42.9 & 65.0\% & 360 & 12.00 & $0.013$$^{*}$ \\
w/o Soft Clustering & 45.5 & [44.7, 46.3] & 43.2 & 65.0\% & 360 & 12.00 & $0.016$$^{*}$ \\
w/o BM25 RRF & 45.3 & [44.3, 46.3] & 43.0 & 65.0\% & 360 & 12.00 & $0.006$$^{**}$ \\
w/o Jaccard Dedup & 45.2 & [44.2, 46.3] & 43.0 & 65.0\% & 780 & 12.00 & $0.019$$^{*}$ \\
Baseline: Naive RAG & 45.0 & [43.8, 46.3] & 42.8 & 65.0\% & 780 & 12.00 & $0.011$$^{*}$ \\
\bottomrule
\end{tabular}
\caption{Component ablation matrix evaluating the impact of individual architectural choices in APEX. $^{**}p < 0.01$, $^{*}p < 0.05$ against Full APEX via paired $t$-test.}
\label{tab:ablation_matrix}
\end{table*}
```
