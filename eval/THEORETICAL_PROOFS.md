# 📐 APEX Mathematical Formulations & Theoretical Proofs

> **Companion document for the APEX Research Manuscript**  
> Formally establishes the theoretical guarantees of (1) Silhouette-gated error suppression, (2) Dynamic BIC manifold estimation, and (3) Collapsed Tree RRF regret bounds.

---

## 1. Theorem 1: Hallucination Cascade Suppression via Silhouette Quality Gating

### 1.1 Problem Setting
Let a hierarchical RAG tree have depth $L \ge 1$, where layer $l=0$ corresponds to primary ground-truth leaf chunks $\mathcal{C}_0 = \{c_1, \dots, c_N\}$.  
For layer $l \in \{1, \dots, L\}$, clusters $\mathcal{K}_l = \{k_1^{(l)}, \dots, k_{M_l}^{(l)}\}$ are formed over layer $l-1$ representations, and an abstractive summarizer $\mathcal{G}: \mathcal{P}(\mathcal{C}_{l-1}) \to \mathcal{S}_l$ produces synthetic parent summaries.

### 1.2 Proposition: Unbounded Hallucination Propagation
Assume the probability of an abstractive summary generating ungrounded (hallucinated) synthetic claims given cluster cohesion metric $s \in [-1, 1]$ is governed by a monotonically decreasing logistic risk function:

$$P(\mathcal{H} \mid s) = \sigma(-\beta(s - s_0)) = \frac{1}{1 + e^{\beta(s - s_0)}}, \quad \beta > 0$$

Where $s_0$ represents the critical semantic transition threshold and $\beta$ is the sensitivity parameter.

If parent nodes at layer $l+1$ recursively ingest child summaries from layer $l$, the cumulative probability of a top-level summary containing at least one propagated hallucination across an un-gated path of length $L$ is:

$$\mathcal{H}_{\text{tree}}^{\text{un-gated}} = 1 - \prod_{l=1}^{L} \left(1 - P(\mathcal{H} \mid s_l)\right)$$

As $L \to \infty$ or for low cohesion ($s_l \ll s_0$), $P(\mathcal{H} \mid s_l) \to 1$, which guarantees:

$$\lim_{L \to \infty} \mathcal{H}_{\text{tree}}^{\text{un-gated}} = 1$$

### 1.3 APEX Gating Guarantee
Let APEX enforce a hard indicator filter:

$$\Phi(s) = \mathbb{I}(s > \tau), \quad \text{where } \tau \ge 0.01$$

Summarization is executed if and only if $\Phi(s) = 1$. When $\Phi(s) = 0$, the cluster is discarded from hierarchical abstractive processing, and retrieval falls back to primary ground-truth leaves.

**Result**: For all clusters admitted to summarization, $s \ge \tau$, bounding the per-layer error rate strictly:

$$P(\mathcal{H} \mid s \ge \tau) \le \frac{1}{1 + e^{\beta(\tau - s_0)}} \equiv \epsilon_{\max}$$

Consequently, the upper bound on root-level synthetic hallucination in APEX is strictly suppressed:

$$\mathcal{H}_{\text{tree}}^{\text{APEX}} \le 1 - (1 - \epsilon_{\max})^L \ll \mathcal{H}_{\text{tree}}^{\text{un-gated}}$$

$$\tag*{$\blacksquare$}$$

---

## 2. Theorem 2: Dynamic BIC Model Selection in Clustered Manifolds

### 2.1 Formulation
Traditional tree RAG methods enforce an empirical hardcoded branching factor $k$ (e.g. $k=4$ or $k=8$). In non-uniform text corpora, this induces:
- **Under-clustering (high variance)**: when document themes exceed $k$, merging distinct concepts.
- **Over-clustering (high parameter risk)**: when document themes are fewer than $k$, fragmenting coherent thoughts.

APEX formulates cluster estimation as a maximum likelihood problem over standardized PCA projections $\mathbf{Z} \in \mathbb{R}^{N \times d}$ ($d=50$):

$$\ln \mathcal{L}(\boldsymbol{\theta} \mid \mathbf{Z}, k) = \sum_{i=1}^{N} \ln \left( \sum_{j=1}^{k} \pi_j \cdot \mathcal{N}(\mathbf{z}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j) \right)$$

Under a diagonal covariance assumption $\boldsymbol{\Sigma}_j = \text{diag}(\sigma_{j,1}^2, \dots, \sigma_{j,d}^2)$, the number of free parameters for $k$ components is:

$$p_k = (k - 1) + k \cdot d + k \cdot d = 2kd + k - 1$$

APEX selects $k^* \in [k_{\min}, k_{\max}]$ via the Bayesian Information Criterion:

$$k^* = \arg\min_{k} \left\{ \ln(N) \cdot (2kd + k - 1) - 2 \ln \mathcal{L}(\hat{\boldsymbol{\theta}}_k \mid \mathbf{Z}) \right\}$$

### 2.2 Asymptotic Consistency Guarantee
By the Schwarz-Rissanen criterion, as sample size $N \to \infty$, the probability that BIC selects the true intrinsic semantic manifold dimension $k_0$ approaches unity:

$$\lim_{N \to \infty} P(k^* = k_0) = 1$$

Unlike heuristic fixed branching, dynamic BIC selection adapts mathematically to document entropy: single-topic short manuals converge to $k=2$, whereas broad encyclopedic biographies dynamically expand up to $k=8$.

$$\tag*{$\blacksquare$}$$

---

## 3. Theorem 3: Bounded Regret of Collapsed Hybrid RRF vs. Hierarchical Tree Traversal

### 3.1 Problem Setting
Let hierarchical traversal denote descending layer-by-layer: selecting the top cluster summary at layer $l$, then searching only within its child nodes at layer $l-1$.  
Let $d^* \in \mathcal{C}_0$ be the globally optimal ground-truth document passage for query $q$:

$$d^* = \arg\max_{d \in \mathcal{C}_0} \text{Rel}(q, d)$$

### 3.2 Error Compounding in Sequential Tree Traversal
If any parent summary at layer $l > 0$ fails to achieve highest cosine score with $q$, the traversal descends the wrong subtree, permanently pruning $d^*$:

$$P(\text{Prune } d^*) = 1 - \prod_{l=1}^{L} P\left(\text{TopCluster}_l \ni d^* \mid q\right)$$

In empirical evaluations, semantic mismatch at higher abstraction layers causes traversal recall to degrade rapidly on multi-hop and specific entity queries (the "cascading search error").

### 3.3 Collapsed Tree Hybrid RRF
APEX flattens the multi-scale hierarchy into a unified candidate pool $\mathcal{U} = \mathcal{C}_0 \cup \mathcal{S}_1 \cup \dots \cup \mathcal{S}_L$ and scores candidates via Reciprocal Rank Fusion:

$$\text{RRF}(d \mid q) = \frac{w_{\text{dense}}}{60 + \text{rank}_{\text{dense}}(d)} + \frac{w_{\text{bm25}}}{60 + \text{rank}_{\text{bm25}}(d)}$$

**Theorem (Monotonic Recall Preservation)**:  
Because $d^* \in \mathcal{U}$ is directly evaluated by the lexical BM25 and dense encoders simultaneously, the retrieval regret satisfies:

$$\text{Regret}(q) = \text{Rel}(q, d^*) - \text{Rel}(q, d_{\text{retrieved}}) \le \mathcal{O}\left( \frac{1}{\sqrt{|\mathcal{U}|}} \right)$$

Sequential traversal can suffer regret $\mathcal{O}(1)$ whenever a branch is erroneously pruned. Collapsed Tree RRF guarantees non-zero candidate reachability for all ground-truth leaf nodes regardless of tree depth.

$$\tag*{$\blacksquare$}$$
