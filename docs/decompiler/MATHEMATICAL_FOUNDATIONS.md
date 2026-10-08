# Mathematical Foundations

## Target-versus-alternatives margin

For target class \(t\) and logits \(z\), VRAVEN uses:

\[
m_t(x)=z_t(x)-\log\sum_{c\ne t}\exp z_c(x)
\]

For binary logits this equals the ordinary target logit difference. It is invariant to adding the same constant to all logits.

## Concept decomposition

For internal representation matrix \(H_l\):

\[
H_l\approx Z_lD_l^\top
\]

where columns of \(D_l\) define candidate concept directions and \(Z_l\) contains sample-level coefficients.

The sparse option optimises a reconstruction term together with sparsity and coherence regularisation:

\[
\mathcal{L}_{concept}=\|H_l-Z_lD_l^\top\|_F^2+
\lambda_1\|Z_l\|_1+\lambda_2\Omega(D_l)
\]

## Concept alignment

For adjacent concept sets, a cost matrix combines direction similarity and activation similarity. Entropic optimal transport estimates a correspondence matrix:

\[
\Pi^*=\arg\min_{\Pi\in\mathcal{U}(a,b)}
\langle C,\Pi\rangle+	au\sum_{ij}\Pi_{ij}(\log\Pi_{ij}-1)
\]

The transport is interpreted as an alignment hypothesis, not proof of semantic identity.

## Intervention effect

For internal concept \(c_g\), declared intervention \(I_g\), and target margin:

\[
\Delta_g(x)=m_t(x)-m_t^{{I_g}}(x)
\]

VRAVEN reports the sample distribution, median, bootstrap interval and sign stability.

## Causal edge effect

For proposed source concept \(i\) and downstream concept \(j\):

\[
E_{i\rightarrow j}=\operatorname{median}_x
\left[c_j(x)-c_j(x;do(c_i=0))\right]
\]

The edge can also record a local derivative screen:

\[
J_{ij}=\frac{\partial c_j}{\partial c_i}
\]

Derivative support does not replace the intervention.

## Programme extraction

Let \(\phi(x)\) be selected concept and interaction terms. VRAVEN fits:

\[
\hat{m}_t(x)=\beta_0+\sum_k\beta_k\tilde{\phi}_k(x)
\]

subject to observational and intervention constraints. For intervention \(I_g\):

\[
\hat{\Delta}_g=\hat{m}_t(x)-\hat{m}_t^{I_g}(x)
\]

The weighted objective is:

\[
\min_{\beta}
\|m-\Phi\beta\|_2^2+
\lambda_I\|\Delta-\Phi_I\beta\|_2^2+
\lambda_1\|\beta\|_1
\]

The first term preserves ordinary behaviour. The second term constrains the programme to reproduce tested internal interventions. The third favours a compact programme.

## Fidelity

VRAVEN reports separate dimensions:

\[
F_{obs}=1-\frac{\sum_i(m_i-\hat{m}_i)^2}{\sum_i(m_i-\bar m)^2}
\]

\[
F_{int}^{sign}=\frac{1}{G}\sum_g\mathbf{1}
\left[\operatorname{sign}(\Delta_g)=\operatorname{sign}(\hat{\Delta}_g)\right]
\]

\[
F_{int}^{corr}=\operatorname{corr}(\Delta,\hat{\Delta})
\]

Counterfactual fidelity is reported through programme and original-model flip agreement. No single combined trust score is created.

## Counterfactual proof

VRAVEN searches over bounded concept replacements:

\[
c^*=\arg\min_{c'}D(c,c')
\]

subject to a programme flip and, when required, an original-model flip. Domain validity is a separate constraint supplied by the application.

## Semantic coverage

Validated semantic coverage and association-only coverage are separate. Causally weighted validated semantic coverage is:

\[
C_{sem}=\sum_g p_g\mathbf{1}[c_g\text{ validated}]
\]

where:

\[
p_g=\frac{|\Delta_g|}{\sum_j|\Delta_j|+\varepsilon}
\]

Association-derived labels do not count as validated grounding.
