# APEX Systems, Compute & Hardware Efficiency Profile

> Generated: 2026-10-27 18:39:20

| Architecture / System | Prompt Tokens | Token Savings | Inference Latency | TTFT | Theoretical FLOPs | Cost / 1M Queries | Indexing (10k tok) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Naive Flat RAG (Lewis et al., 2020) | 1158.7 tok | Baseline | 12.85 s | 16.39 s | 2402 GFLOPs | $199.31 | 1.8 s |
| RAPTOR Tree RAG (Sarthi et al., ICLR 2024) | 908.0 tok | -21.6% vs. Naive | 18.19 s | 16.69 s | 1896 GFLOPs | $160.20 | 42.5 s |
| **APEX (Ours)** | 363.5 tok | -60.0% vs. RAPTOR | 9.61 s | 9.86 s | 803 GFLOPs | $77.33 | 14.2 s |

---

## LaTeX Source Code (`tab:systems_efficiency`)

```latex
\begin{table*}[t]
\centering
\small
\begin{tabular}{l c c c c c c}
\toprule
\textbf{System Architecture} & \textbf{Context Tokens} & \textbf{Latency (s)} & \textbf{TTFT (s)} & \textbf{GFLOPs/Query} & \textbf{Cost / 1M Queries} & \textbf{Indexing Time} \\
\midrule
Naive Flat RAG (Lewis et al., 2020) & 1158.7 & 12.85 & 16.39 & 2402 & \$199.31 & 1.8 s \\
RAPTOR Tree RAG (Sarthi et al., ICLR 2024) & 908.0 & 18.19 & 16.69 & 1896 & \$160.20 & 42.5 s \\
\textbf{APEX (Ours)} & \textbf{363.5} & \textbf{9.61} & \textbf{9.86} & \textbf{803} & \textbf{\$77.33} & 14.2 s \\
\bottomrule
\end{tabular}
\caption{Systems and compute efficiency profile across retrieval architectures using a 1B parameter backbone. Theoretical FLOPs denote $2 \times N_{\text{params}} \times T_{\text{total}}$. Cost is projected at standard cloud serving rates (\$0.15/M input, \$0.60/M output tokens).}
\label{tab:systems_efficiency}
\end{table*}
```
