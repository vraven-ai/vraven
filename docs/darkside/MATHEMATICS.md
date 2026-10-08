# Mathematical definitions

This document defines the quantities implemented by VRAVEN DarkSide. It also states the limits of each quantity. Composite indices are transparent ranking devices unless calibration evidence is supplied.

## 1. Approximate ε-sufficient decision circuit

Let the declared candidate internal groups be \(\mathcal G\), the selected target score be \(s_t(x)\), and \(I_R\) denote an intervention that replaces all groups \(R\subseteq\mathcal G\) with the declared baseline. A retained set \(C=\mathcal G\setminus R\) is accepted when

\[
\hat y(I_R(x))=\hat y(x)
\]

and

\[
\left|s_t(I_R(x))-s_t(x)\right|\leq\varepsilon.
\]

The implementation measures individual effects, sorts candidates by absolute effect, and applies greedy backward elimination. It verifies every accepted removal with an actual forward intervention. The returned core is approximate and conditional on the candidate universe, grouping, sample, baseline and order.

The individual score drop for group \(g\) is

\[
d(g)=s_t(x)-s_t(I_g(x)).
\]

The final conditional necessity audit tests each retained group after all accepted removals have already been applied.

## 2. Shortcut dependence

For a declared input group \(g\), local intervention effect is

\[
\Delta_g=s_t(x)-s_t(I_g(x)).
\]

First-order attribution dominance is

\[
D_g=
\frac{\sum_{i\in g}\left|x_i\,\partial s_t/\partial x_i\right|}
{\sum_j\left|x_j\,\partial s_t/\partial x_j\right|}.
\]

The supplied shortcut suspicion index combines narrow input coverage, relative intervention effect and attribution dominance only for within-audit ranking. It has no universal decision threshold. A shortcut claim also requires domain evidence that the cue is semantically irrelevant, unstable or causally inappropriate.

## 3. Prediction depth

For each layer \(l\), a local k-nearest-neighbour probe predicts a class from the reference representation. For query sample \(i\), let \(\hat y_{i,l}\) be the probe prediction at layer \(l\), and let \(\hat y_{i,L}\) be the final model prediction. The prediction depth is the earliest layer that agrees with the final prediction and remains final-consistent through all later tested layers:

\[
d_i=
\min\left\{l:
\hat y_{i,k}=\hat y_{i,L}\;\forall k\geq l
\right\}.
\]

This is a probe-based localisation of decodable class structure. It does not prove that the model causally used the probe boundary.

## 4. Class-conditional effect size

For unit \(j\) in two cohorts, Hedges' \(g\) is

\[
g_j=J\frac{\bar a_j-\bar b_j}{s_{p,j}},
\qquad
J=1-\frac{3}{4(n_a+n_b-2)-1}.
\]

Energy distance and RBF maximum mean discrepancy compare the full empirical distributions. For random vectors \(X\sim P\) and \(Y\sim Q\), the population energy distance is

\[
\mathcal E(P,Q)
=
2\mathbb E\lVert X-Y\rVert
-
\mathbb E\lVert X-X'\rVert
-
\mathbb E\lVert Y-Y'\rVert.
\]

The implemented finite-sample value is non-negative up to numerical tolerance.

## 5. Centred kernel alignment

For aligned representation matrices \(X\) and \(Y\), with centred linear Gram matrices \(K_c\) and \(L_c\),

\[
\operatorname{CKA}(X,Y)=
\frac{\langle K_c,L_c\rangle_F}
{\lVert K_c\rVert_F\lVert L_c\rVert_F}.
\]

Rows must refer to the same ordered samples. CKA is not calculated between unrelated class members.

## 6. Representation drift

For each aligned layer, VRAVEN retains raw metrics and also constructs bounded, dimensionless components:

\[
d_{\mathrm{CKA}}=1-\operatorname{CKA}(X,Y),
\]

\[
d_{\mathrm{Proc}}\in[0,1],
\]

where \(d_{\mathrm{Proc}}\) is the normalised orthogonal Procrustes residual,

\[
d_{\mathrm{MMD}}=
\operatorname{clip}\left(\frac{\widehat{\mathrm{MMD}}^2}{2},0,1\right),
\]

and

\[
d_{\mathrm{SW}}=
\frac{\widehat W_{\mathrm{sliced}}/b}
{1+\widehat W_{\mathrm{sliced}}/b},
\]

where \(b\) is the median pairwise-distance bandwidth calculated from the two compared representations.

The transparent drift ranking index is

\[
D_l=
0.35d_{\mathrm{CKA}}
+0.25d_{\mathrm{Proc}}
+0.20d_{\mathrm{MMD}}
+0.20d_{\mathrm{SW}}.
\]

Therefore \(D_l\in[0,1]\). It is not a calibrated probability of failure.

## 7. Stability geometry

For paired clean and perturbed representations, VRAVEN reports paired cosine similarity, relative \(\ell_2\) change, CKA, k-neighbourhood overlap and principal-angle distance. A transparent composite ranks possible collapse points, but all components remain visible.

For paired rows \(x_i,y_i\), relative change is

\[
r_i=
\frac{\lVert y_i-x_i\rVert_2}
{\max(\lVert x_i\rVert_2,\epsilon)}.
\]

Neighbourhood overlap compares the k-nearest-neighbour index sets in the two aligned spaces. Principal-angle distance compares leading subspaces. These measurements diagnose different geometric changes and should not be collapsed into an unexplained single score.

## 8. Cross-method disagreement

For vectors on the same indexed surface, VRAVEN reports Spearman correlation, Kendall \(\tau_b\), top-k Jaccard overlap and sign agreement. Per element, rank dispersion and sign entropy are

\[
H_i=-\frac{1}{\log 3}
\sum_{c\in\{+, -, 0\}}p_{ic}\log p_{ic}.
\]

Agreement is not evidence of correctness. Disagreement can expose method sensitivity, numerical approximation or genuinely different formal targets.

## 9. Explanation uncertainty

Repeated explanations form \(A\in\mathbb R^{R\times d}\). VRAVEN estimates per-element mean, bootstrap interval, sign probability, top-k selection probability and rank standard deviation.

For element \(j\), empirical top-k selection probability is

\[
\hat p_j^{(k)}=
\frac{1}{R}
\sum_{r=1}^{R}
\mathbb 1\left[j\in\operatorname{TopK}(|A_r|)\right].
\]

The intervals cover only the declared source of variation, such as retraining seed, stochastic explainer run, background resampling or controlled input perturbation.

## 10. Capacity

For singular-value power \(p_i=\sigma_i^2/\sum_j\sigma_j^2\), effective rank is

\[
r_{\mathrm{eff}}=
\exp\left(-\sum_i p_i\log p_i\right).
\]

Stable rank is

\[
r_{\mathrm{stable}}=
\frac{\lVert X\rVert_F^2}{\lVert X\rVert_2^2}.
\]

Participation ratio is

\[
r_{\mathrm{PR}}=
\frac{\left(\sum_i\sigma_i^2\right)^2}
{\sum_i\sigma_i^4}.
\]

These describe the geometry of the stored representation. They do not equal the exact number of removable parameters.

## 11. Adversarial or evasion rerouting

For aligned clean and altered representations at layer \(l\), the mean paired representation cosine is

\[
C_l=
\frac{1}{n}\sum_{i=1}^{n}
\frac{x_{i,l}^{\top}y_{i,l}}
{\lVert x_{i,l}\rVert_2\lVert y_{i,l}\rVert_2}.
\]

The basic reroute index is

\[
R_l=1-C_l.
\]

VRAVEN also records representation CKA, neighbourhood overlap, unit influence cosine and sign-flip fraction when available. The input \(\ell_0\) and \(\ell_2\) changes may be supplied separately. Rerouting does not prove malicious construction.

## 12. Cost–evidence trade-off

For layer \(l\), mean measured latency is \(c_l\) and mean evidence signal is \(e_l\). The evidence rate is

\[
\eta_l=\frac{e_l}{\max(c_l,\epsilon)}.
\]

A layer is Pareto efficient when no other tested layer has both no greater cost and no lower evidence, with at least one strict improvement. The Pareto frontier is a diagnostic. It does not prove that a dominated layer can be removed because downstream representations may depend on it.

## 13. Circuit multiplicity and empirical stable core

Repeated circuit discovery produces binary masks

\[
M\in\{0,1\}^{R\times d},
\]

where rows are discovery runs and columns are aligned components. Component inclusion probability is

\[
\hat p_j=\frac{1}{R}\sum_{r=1}^{R}M_{rj}.
\]

VRAVEN uses a Wilson score interval for \(p_j\). A component enters the empirical stable core only when the lower interval bound is at least the declared threshold \(\tau\).

Pairwise circuit agreement is the Jaccard index

\[
J_{rs}=
\frac{|C_r\cap C_s|}{|C_r\cup C_s|}.
\]

The multiplicity index is

\[
\mathcal M=1-\frac{2}{R(R-1)}
\sum_{r<s}J_{rs}.
\]

Component inclusion entropy is

\[
h_j=-\hat p_j\log_2\hat p_j
-(1-\hat p_j)\log_2(1-\hat p_j).
\]

This is an empirical repeated-discovery analysis, not a formal certificate over an input domain or dataset edit neighbourhood.

## 14. Pairwise causal interaction

For target-score drops \(d(A)\), \(d(B)\), and \(d(A,B)\),

\[
\operatorname{INT}(A,B)=d(A,B)-d(A)-d(B).
\]

Positive values are super-additive under the declared intervention. Negative values indicate redundancy or compensation. Higher-order interactions are not recovered by pairwise analysis.

## 15. Activation mediation and repair

For clean score \(s_c\), corrupted score \(s_k\), and a corrupted run patched with the clean activation at layer \(l\), \(s_{k\leftarrow c,l}\),

\[
R_l=s_{k\leftarrow c,l}-s_k,
\qquad
\rho_l=
\frac{R_l}{s_c-s_k}.
\]

Optional reverse noising places the corrupted activation into the clean run and measures target-score damage. The patching recovery is not automatically a pure natural indirect effect. Interactions and off-manifold hybrid states can affect it.

## 16. H0 persistence and optional H1 summary

For a Vietoris–Rips filtration of an empirical point cloud, H0 component death times equal the edge weights of its minimum spanning tree. VRAVEN computes these death times exactly and compares their sorted empirical distributions with the one-dimensional Wasserstein distance.

Optional H1 analysis, when `ripser` is installed, compares quantiles of finite H1 lifetimes. It is deliberately labelled a lifetime-distribution distance, not a complete bottleneck or Wasserstein distance between persistence diagrams.

## 17. Mechanism transfer

For aligned intervention-effect vectors \(u\) and \(v\), VRAVEN reports signed cosine similarity

\[
\cos(u,v)=
\frac{u^{\top}v}
{\lVert u\rVert_2\lVert v\rVert_2},
\]

normalised \(\ell_1\) distance, sign agreement and top-k overlap. The stable transfer core is the intersection of the declared top-k components across all conditions. Component identity must remain genuinely aligned.

## 18. High-confidence internal anomaly

With reference fingerprint mean \(\mu\) and shrinkage precision \(\Theta\),

\[
D_M(z)=
\sqrt{(z-\mu)^\top\Theta(z-\mu)}.
\]

The query score is converted to an empirical reference percentile. A high-confidence output beyond the declared percentile threshold is a review signal. It is not proof of an attack or out-of-distribution input.

## 19. Claim hierarchy

The extension uses the following evidence hierarchy:

1. **Descriptive:** activation energy, sparsity, timing and rank.
2. **Associational:** CKA, MMD, probes, class separation and method agreement.
3. **Paired diagnostic:** clean-versus-perturbed path change and mechanism transfer.
4. **Intervention-based:** ablation, patching and pairwise interaction under a declared baseline.
5. **Formally certified:** not currently implemented by this extension.

No lower level should be described using a stronger causal claim.
