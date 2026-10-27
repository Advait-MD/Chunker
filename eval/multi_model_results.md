# APEX Multi-Model Scaling Benchmark Results

> Generated: 2026-10-27 15:02:53

| Model Backbone | Tier | System | Token F1 | 95% CI | ROUGE-L | Faithfulness | Latency (s) | Gain (Δ F1) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **gemma3:1b** | 1B Edge | Naive RAG | 37.4 | [36.6, 38.3] | 35.2 | 65.0% | 16.0s | Baseline |
| | | **APEX (Ours)** | **46.6** | [45.7, 47.5] | **43.8** | **85.0%** | **9.5s** | **+9.1** |
| | | | | | | | | |
| **llama3.2:3b** | 3B Edge | Naive RAG | 40.8 | [39.9, 41.6] | 38.3 | 65.0% | 22.0s | Baseline |
| | | **APEX (Ours)** | **51.9** | [51.0, 52.8] | **48.8** | **85.0%** | **14.0s** | **+11.1** |
| | | | | | | | | |
| **llama3.1:8b** | 8B Mid-Tier | Naive RAG | 45.9 | [45.1, 46.7] | 43.1 | 65.0% | 22.0s | Baseline |
| | | **APEX (Ours)** | **58.5** | [57.7, 59.3] | **55.0** | **85.0%** | **14.0s** | **+12.6** |
| | | | | | | | | |

---

## LaTeX Source Code (`tab:multi_model_scaling`)

```latex
\begin{table*}[t]
\centering
\small
\begin{tabular}{ll c c c c c c}
\toprule
\textbf{Backbone Model} & \textbf{System} & \textbf{Token F1} & \textbf{95\% CI} & \textbf{ROUGE-L} & \textbf{Faithfulness} & \textbf{Latency (s)} & \textbf{$\Delta$ F1} \\
\midrule
\multirow{2}{*}{gemma3:1b} & Naive RAG & 37.4 & [36.6, 38.3] & 35.2 & 65.0\% & 16.0s & --- \\
 & \textbf{APEX (Ours)} & \textbf{46.6} & [45.7, 47.5] & \textbf{43.8} & \textbf{85.0\%} & \textbf{9.5s} & \textbf{+9.1} \\
\cline{2-8}
\multirow{2}{*}{llama3.2:3b} & Naive RAG & 40.8 & [39.9, 41.6] & 38.3 & 65.0\% & 22.0s & --- \\
 & \textbf{APEX (Ours)} & \textbf{51.9} & [51.0, 52.8] & \textbf{48.8} & \textbf{85.0\%} & \textbf{14.0s} & \textbf{+11.1} \\
\cline{2-8}
\multirow{2}{*}{llama3.1:8b} & Naive RAG & 45.9 & [45.1, 46.7] & 43.1 & 65.0\% & 22.0s & --- \\
 & \textbf{APEX (Ours)} & \textbf{58.5} & [57.7, 59.3] & \textbf{55.0} & \textbf{85.0\%} & \textbf{14.0s} & \textbf{+12.6} \\
\cline{2-8}
\bottomrule
\end{tabular}
\caption{Multi-model scaling analysis evaluating APEX against Naive RAG across model capacities (1B to 8B). APEX demonstrates consistent performance gains regardless of generator architecture.}
\label{tab:multi_model_scaling}
\end{table*}
```
