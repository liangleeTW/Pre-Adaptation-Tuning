# Model Expansion Plan: Proprioceptive Tuning & Measurement Noise Modulation

**Version:** 2.0
**Date:** 2026-01-06
**Status:** Planning phase - expanding beyond M0/M1/M2

---

## Executive Summary

**Research Question:** Does proprioceptive tuning (Δπ) during pre-adaptation modulate measurement noise (R) in subsequent visuomotor adaptation?

**Current Status:** M0/M1/M2 show modest effects; results not strong enough for publication. M2's tanh over-saturates given small observed Δπ. Need theoretically-grounded, mechanistically-distinct models.

**Strategy:** Expand to test (1) alternative measurement noise modulations (exponential, cue-combination), (2) process noise modulation (meta-learning), and (3) joint mechanisms (R+Q). All models must be familiar to reviewers and grounded in established literature.

---

## 🔥 EMPIRICAL RESULTS (2026-01-06)

### Full Dataset Fit: N=69 subjects, 4 chains × 2000 samples, Δlogπ metric

| Model     | WAIC   | LOO    | ΔWAIC from M0 | Converged? | Winner? |
|-----------|--------|--------|---------------|------------|---------|
| **M0**    | 46983  | 46983  | 0 (baseline)  | ✅ Yes      | ❌      |
| **M1**    | 44688  | 44581  | **-2295**     | ✅ Yes      | ❌      |
| **M1-exp**| 44425  | 44360  | **-2558**     | ✅ Yes      | ✅ **WINNER** |
| **M-Q**   | 57421  | 44608  | +10438        | ❌ **NO** (Rhat=1.53, ESS=7) | ❌ |
| **M-hybrid**| 48160 | 43303 | +1177         | ❌ **NO** (Rhat=1.53, ESS=7) | ❌ |
| **M2**    | 45715  | 45640  | -1268         | ✅ Yes      | ❌      |

### Critical Findings:

#### ✅ **M1-exp is the clear winner among converged models**
- **ΔWAIC = -263** relative to M1 (moderate improvement)
- **ΔWAIC = -2558** relative to M0 (strong improvement; tuning clearly matters)
- **Converged perfectly:** Rhat < 1.01, ESS_bulk > 10,000
- **Parameter estimates (β):** EC=0.97, EO+=0.91, EO-=0.92
- **Interpretation:** All groups show **positive β**: higher Δlogπ → higher R → lower Kalman gain
  - **Cognitive meaning:** Sharpening proprioception during pre-adaptation makes the system *distrust visual errors more* during adaptation, slowing learning
  - **Alternative interpretation:** Multicollinearity with R_post1; β may just rescale baseline without adding mechanistic insight

#### ❌ **M-Q and M-hybrid FAILED catastrophically**
- **Non-convergence:** Rhat = 1.53 (should be < 1.01), ESS_bulk = 7 (should be > 400)
- **Unreliable posteriors:** Chains did not mix; parameter estimates are meaningless
- **Root cause:** Prior Normal(0, 1) on β_q allows Q to explode by orders of magnitude
  - With Q_baseline = 1e-4 and β_q ≈ 10, Q becomes 7× larger → Kalman filter unstable
  - In M-hybrid, β_q ranged from -18.69 to +9.44 (EO+ group had negative β_q, making Q collapse)
- **Huge WAIC-LOO discrepancies:** 12,813 for M-Q; 4,857 for M-hybrid (indicates severe model misspecification)

#### ⚠️ **M1 is solid but outperformed by M1-exp**
- Strong improvement over M0 (ΔWAIC = -2295)
- β estimates similar to M1-exp but additive link
- M1-exp preferred due to better WAIC and multiplicative interpretation

#### 🤔 **M2 (tanh) is viable but offers no advantage**
- Converged well (Rhat < 1.01)
- λ saturated at negative values (-0.64 to -0.73), confirming over-saturation issue
- WAIC worse than M1 and M1-exp; added complexity not justified

### Revised Conclusions:

1. **Process noise (Q) modulation is empirically unidentifiable** with current data/priors
   - Theoretical appeal does not translate to stable inference
   - Interaction between Q and R creates non-identifiability when both vary

2. **Measurement noise (R) modulation via M1-exp is the most parsimonious explanation**
   - Exponential link provides numerical stability
   - Effect size is modest but reliable (β ≈ 0.9-1.0 across groups)

3. **Effect is counter-intuitive:** Proprioceptive sharpening → *slower* adaptation
   - Possible mechanisms:
     - Source estimation: sharper proprioception shifts blame away from visual errors
     - Sensory reweighting: increased proprioceptive reliability → decreased visual error trust
     - Spurious correlation: β may track individual differences not causally related to Δπ

4. **Next steps:**
   - Posterior predictive checks for M1-exp (validate fit quality)
   - Residual analysis (check for systematic misfit)
   - Parameter correlation plots (assess multicollinearity with R_post1)
   - Consider simplified models: fixed β across groups, test if group differences are real

---

## Current Models & Results

### Data Context
- **Δπ range:** 95% within ±0.72; group means EC≈0.20, EO+≈0.10, EO−≈0.06
- **R_post1 range:** ~0.55–5.2 (measurement noise variance)
- **Metric choice:** Δlogπ = log(precision_post1/precision_pre) gives better WAIC than Δπ for multiplicative models

### M0: Baseline (No Modulation)
```
R = R_post1
```
**Purpose:** Null model; tests whether tuning matters at all
**Parameters:** None (R fixed to proprioceptive baseline)
**Interpretation:** Kalman learner with constant measurement noise

### M1: Linear Additive
```
R = R_post1 + β·Δπ  (or Δlogπ)
```
**Purpose:** Simplest monotonic modulation; one parameter per group
**Parameters:** β (slope)
- β > 0: Higher Δπ → higher R → lower Kalman gain (distrust visual errors)
- β < 0: Higher Δπ → lower R → higher gain (trust visual errors)

**Issues:**
- Multicollinearity with R_post1 (β may just rescale baseline)
- Can produce negative R if β·Δπ < -R_post1 (requires clipping)

### M2: Tanh Multiplicative
```
R = R_post1 · (1 - λ·tanh(Δπ))
```
**Purpose:** Bounded modulation with asymmetric saturation
**Parameters:** λ (strength, constrained to [-1, 1] via tanh transformation)

**Issues:**
- **Over-saturates:** With Δπ ∈ [-0.42, 1.23], tanh(Δπ) ∈ [-0.4, 0.84]; most data in near-linear regime
- Saturations occurred in ~20-40% of posterior samples (λ posteriors drifted to ±1)
- Added complexity without benefit; use simpler bounded form if needed

**Conclusion:** M0/M1/M2 provide weak separation between groups; need mechanistically richer models.

---

## Proposed Model Expansion: Critical Evaluation

### Tier 1: Recommended Core Models

#### **M1-exp: Exponential / Log-Link** ✅ **STRONG CANDIDATE**
```
R = R_post1 · exp(β · Δlogπ)
```

**Cognitive Meaning:**
Each unit increase in log-precision scales R by a constant percentage. Multiplicative gain modulation of measurement noise.

**Parameter Effects:**
- β controls elasticity: d ln R / d(Δlogπ)
- Small |β| ≈ linear M1; guarantees R > 0 without clipping
- β > 0: Sharper proprioception → higher R → lower Kalman gain
- β < 0: Sharper proprioception → lower R → higher gain

**Strengths:**
- **Well-established:** Log-link is standard in GLMs; familiar to reviewers
- **Guarantees R > 0:** No artificial bounds needed
- **Already implemented** in codebase (model.py:27, fit script:114-117)
- **Natural with Δlogπ:** Proportional change maps to proportional scaling

**Weaknesses:**
- Still single-parameter per group; can't capture asymmetry
- Multicollinearity with R_post1 remains
- Risk of over-inflation if β > 1 and subject has Δlogπ > 1

**Expected Results:**
With Δlogπ ∈ [-0.2, 0.5] (typical), exp keeps curvature mild. EO− (highest R_post1) should show largest absolute R shifts. If β > 0, group ordering widens; if β < 0, ordering compresses.

**Verdict:** **Keep as primary alternative to M1.** Natural progression testing additive vs multiplicative modulation.

---

#### **M-Q: Process Noise Modulation (Meta-Learning)** ✅ **STRONG CANDIDATE**
```
Q = Q_baseline · exp(β_Q · Δlogπ)
R = R_post1  (fixed)
```

**Cognitive Meaning:**
Tuning affects **internal model volatility** (how fast the state estimate updates), not sensory noise. Aligns with meta-learning frameworks: sharper proprioception during pre-adaptation signals higher environmental volatility → faster subsequent adaptation.

**Parameter Effects:**
- β_Q > 0: Higher Δπ → higher Q → faster learning (higher effective gain even with fixed R)
- β_Q < 0: Higher Δπ → lower Q → slower learning (more conservative updates)
- Mechanistically **distinct from R modulation:** Q affects update speed; R affects error weighting

**Theoretical Grounding:**
- **Behrens et al. (2007):** Volatility-driven learning rate adaptation
- **Nassar et al. (2010, 2012):** Uncertainty-guided learning; change-point detection
- **Meta-learning:** Pre-adaptation as contextual cue for subsequent learning dynamics

**Strengths:**
- **Novel mechanism:** Tests alternative to measurement noise story
- **Strong theoretical basis:** Reviewers familiar with volatility/meta-learning literature
- **Testable predictions:**
  - If β_Q > 0: EC (highest Δπ) should have steeper initial slopes but same asymptote
  - If β_Q < 0: EO groups learn slower despite similar R
- **Separates from R effects:** Can test M-Q vs M1-exp to identify dominant mechanism

**Implementation:**
```python
def compute_r_mq(q_baseline: float, beta_q: float, delta_logpi: float) -> tuple[float, float]:
    q = q_baseline * math.exp(beta_q * delta_logpi)
    return q

# In Kalman filter:
p_pred = A * p * A + q_modulated  # Use Q from above
```

**Expected Results:**
If pre-adaptation tunes volatility expectations, groups should differ in early learning rate slopes even with matched R. WAIC comparison to M1-exp identifies whether noise or volatility dominates.

**Verdict:** **High priority.** Mechanistically distinct, theoretically grounded, gives alternative explanation if M1/M1-exp are weak.

---

#### **M-hybrid: Joint R+Q Modulation** ✅ **STRONG CANDIDATE**
```
R = R_post1 · exp(β_R · Δlogπ)
Q = Q_baseline · exp(β_Q · Δlogπ)
```

**Cognitive Meaning:**
Tuning affects **both sensory reliability and internal model volatility**. Tests whether pre-adaptation operates on multiple levels: recalibrating both "how much to trust errors" and "how fast the world changes."

**Parameter Effects:**
- β_R dominates: Measurement noise story (sensory weighting)
- β_Q dominates: Meta-learning story (volatility expectation)
- Both nonzero: Multi-level tuning; stronger claim about adaptation sophistication

**Theoretical Grounding:**
- **Körding & Wolpert (2004):** Bayesian learners adjust both sensory and state uncertainty
- **Wei & Körding (2009):** Sensory uncertainty affects learning rates via multiple pathways

**Strengths:**
- **Most general model:** Nests M1-exp and M-Q as special cases
- **Disentangles mechanisms:** Posterior correlation between β_R and β_Q reveals trade-offs
- **Addresses reviewer concern:** "Why assume only R changes? Could be Q or both."

**Weaknesses:**
- **Two parameters per group:** Risk of overfitting if N small or Δπ variance low
- **Identifiability:** If β_R and β_Q are strongly correlated, model is effectively one-dimensional
- **Interpretation complexity:** Harder to communicate if both are nonzero with similar magnitudes

**Expected Results:**
If posteriors show β_R ≠ 0 but β_Q ≈ 0 (or vice versa), identifies dominant mechanism. If both nonzero and uncorrelated, compelling evidence for multi-level tuning. If correlated, suggests one is sufficient.

**Verdict:** **Include as comprehensive test.** If identifiable, strongest mechanistic claim. If not, reveals that simpler model (M1-exp or M-Q alone) is sufficient.

---

### Tier 2: Conditional Models (Test if Tier 1 Weak)

#### **M3-cc: Cue Combination** ⚠️ **CONDITIONAL - IDENTIFIABILITY CONCERNS**
```
R = 1 / (π_vis + π_prop)
π_prop = (1/R_post1) + Δπ  [or use Δlogπ with log-space addition]
```

**Cognitive Meaning:**
Measurement noise is the joint variance of vision and proprioception. Tuning changes only the proprioceptive channel. Directly implements Bayesian cue combination.

**Parameter Effects:**
- π_vis (visual precision): Free parameter, likely weakly identified without independent manipulation
- Larger Δπ → higher π_prop → lower R (higher combined precision → lower noise)
- Group differences emerge via π_vis and baseline π_prop

**Theoretical Grounding:**
- **Ernst & Banks (2002):** Cross-modal cue combination
- **Körding & Wolpert (2004):** Sensorimotor cue integration
- **van Beers et al. (1999):** Vision-proprioception weighting

**Strengths:**
- **Strongest theoretical grounding:** Reviewers will immediately recognize framework
- **Mechanistically clear:** Separates visual vs proprioceptive channels
- **Testable with independent data:** If you have visual-only noise estimates, can fix/constrain π_vis

**Critical Weaknesses:**
- **Identifiability crisis:** π_vis is free parameter not independently measured. Will trade off with R_post1 and Δπ effects.
- **Variance soaking:** π_vis can absorb group differences, making Δπ effects vanish
- **Negativity risk:** If Δπ < -(1/R_post1), π_prop becomes negative (need strict bounds)
- **Prior sensitivity:** Results may depend heavily on π_vis prior choice

**Rescue Strategies:**
1. **Fix π_vis to literature values:** Search visual psychophysics papers for reaching/pointing visual noise estimates; fix π_vis per group
2. **Sensitivity analysis:** Fit with π_vis ∈ {0.5, 1.0, 2.0}; show Δπ effects robust across range
3. **Hierarchical prior:** π_vis ~ Gamma(α, β) with tight hyperpriors from literature

**Implementation:**
```python
def compute_r_m3cc(r_post1: float, delta_pi: float, pi_vis: float) -> float:
    pi_prop = (1.0 / r_post1) + delta_pi
    if pi_prop <= 0:
        pi_prop = 1e-8  # Safety bound
    r = 1.0 / (pi_vis + pi_prop)
    return max(r, 1e-8)
```

**Verdict:** **Test only if you can justify π_vis.** Strong theoretical appeal, but high identifiability risk. Likely better as sensitivity analysis than primary model. Reviewers will ask: "How do you know π_vis? Why not just M1-exp?"

---

#### **M4-joint: Shared Latent for R and Plateau** ⚠️ **CONDITIONAL - CONFOUNDING RISK**
```
R = R_post1 · exp(β · Δlogπ)
b = b_0 + γ · Δlogπ  (plateau bias)
```

**Cognitive Meaning:**
Same proprioceptive change affects both error weighting (R) and residual bias (b). Captures idea that tuning alters both noise and systematic offset in sensory map.

**Parameter Effects:**
- β: As in M1-exp (modulates R)
- γ: Maps Δlogπ to plateau shift
  - γ > 0: Sharper proprioception → larger residual offset
  - γ < 0: Sharper proprioception → smaller plateau (better washout)

**Strengths:**
- **Tests dual outcome:** Tuning might affect late-trial behavior (plateau) independently of early learning
- **Could explain group differences in asymptotes:** If EC/EO groups show different plateaus

**Critical Weaknesses:**
- **Parameter confounding:** β and γ can trade off. Changes in b can mimic changes in R (flatter/steeper approach to same asymptote).
- **Two outcomes, one driver:** Forces coupling; if true process has independent R and b mechanisms, estimates will be biased
- **Noisy plateaus:** Late-trial errors often have high variance (subject disengagement, strategy shifts); Δlogπ → b might overfit noise
- **Model comparison difficulty:** WAIC will be hard to compare to M1-exp (different data targets)

**Alternative Approach:**
- **Test separately:** Fit M1-exp for R; then test post-hoc regression `b ~ Δlogπ` on plateau estimates. If both significant, combine into M4-joint.
- **Null expectation:** If Kalman model is correct, plateau should be b ≈ 0 (or constant per group); Δlogπ should not affect it.

**Verdict:** **Low priority.** Test only if (1) you observe clear Δlogπ-plateau correlation in exploratory plots, AND (2) β and γ posteriors are uncorrelated in M4-joint. Otherwise, likely confounded or overfitting noise.

---

### Tier 3: Not Recommended

#### **M1-asym: Signed Linear** ❌ **SKIP - WEAK IDENTIFIABILITY**
```
R = R_post1 · (1 + β_pos·max(Δπ,0) + β_neg·min(Δπ,0))
```

**Cognitive Meaning:**
Improvements vs decrements in proprioceptive precision have different effects (e.g., gain vs loss asymmetry in sensory weighting).

**Critical Issues:**
- **Data sparsity:** EO− and EO+ have narrow negative Δπ tails; most mass near 0. Insufficient data to identify two slopes.
- **No strong theory:** Why would gains/losses differ? Need mechanistic story (e.g., loss aversion in cue weighting), not just "flexibility."
- **Clipping artifacts:** Need 1 + term > 0; constraints distort gradients, complicate inference
- **Overfitting risk:** With modest Δπ spread, β_pos and β_neg posteriors likely collapse or fit noise

**Verdict:** **Skip.** Unless you see clear asymmetry in Δπ > 0 vs Δπ < 0 subsets in exploratory plots, this will look like data-dredging to reviewers.

---

#### **M5-gated: Error-Magnitude Dependent** ❌ **SKIP - TOO COMPLEX**
```
R_t = R_post1 · (1 + β · Δlogπ · g(|e_t|))
g(|e|) = 1/(1 + |e|/c)  (gating function)
```

**Cognitive Meaning:**
Tuning effects matter most early when errors are large; gating fades as errors shrink. Mirrors context-dependent source estimation.

**Critical Issues:**
- **High complexity:** Trial-level time-varying R + gating function with parameter c
- **Identifiability:** c and β trade off; weakly identified unless early error variance is large
- **Slow inference:** Requires loop over subjects × trials; much slower than vectorized M0-M2
- **WAIC incomparability:** Likelihood structure differs from other models
- **Weak hypothesis:** "Tuning matters more when errors are large" is post-hoc testable via residual analysis, not a mechanistic model

**Alternative:**
- **Post-hoc test:** Fit M1-exp; then test whether residuals (observed - predicted) correlate with |e_t| × Δlogπ interaction. If yes, suggests gating; if no, M1-exp sufficient.

**Verdict:** **Skip for main paper.** This is a follow-up exploratory analysis, not a core model. Adds complexity without clear mechanistic motivation.

---

## Recommended Model Suite for Paper (UPDATED POST-FITTING)

### Main Text: Core Set (REVISED)
1. **M0:** No modulation (baseline) - ✅ **Required for comparison**
2. **M1:** Linear additive (R = R_post1 + β·Δlogπ) - ✅ **Include** (strong baseline)
3. **M1-exp:** Exponential (R = R_post1 · exp(β·Δlogπ)) - ✅ **WINNER** (best WAIC, converged)
4. ~~**M-Q:** Process noise modulation~~ - ❌ **ABANDON** (failed convergence)
5. ~~**M-hybrid:** Joint R+Q~~ - ❌ **ABANDON** (failed convergence)

### Rationale (REVISED):
- **M0:** Establishes baseline; shows tuning matters (ΔWAIC = -2558 to M1-exp)
- **M1:** Demonstrates linear effect; serves as simpler alternative to M1-exp
- **M1-exp:** Winner; provides multiplicative scaling with numerical stability

### Supplementary Material:
- **M2:** Tanh (show saturation issue as methodological lesson)
- **M-Q/M-hybrid failure analysis:** Document non-convergence, explain why Q modulation failed (identifiability, prior sensitivity)
- Sensitivity analyses:
  - Δπ vs Δlogπ metric (already done; logpi wins)
  - Fixed vs group-specific β (test if EC/EO+/EO- truly differ)
  - Restricted Δπ ranges (robustness check)

### What to Report:
1. **Main text:** M0, M1, M1-exp with WAIC table, parameter estimates, posterior predictive plots
2. **Supplement:** M2 (saturated), attempted M-Q/M-hybrid (explain failure as cautionary tale)
3. **Discussion:** Address counter-intuitive positive β (slower adaptation with sharper proprioception); alternative explanations

---

## Additional Parameters & Indices to Consider

Beyond Δπ and R_post1, other data features could be used as predictors:

### From Proprioceptive Data:
1. **Variability of post-test 1 (σ_post1):**
   - Higher variability → lower reliability even with same mean precision
   - Could predict different R for same Δπ
   - Implementation: `R = f(R_post1, Δπ, σ_post1)`

2. **Change in variability (Δσ = σ_post1 - σ_pre):**
   - Tests whether noise change (not just mean shift) affects adaptation
   - Prediction: Larger Δσ → higher R (less reliable even if mean precision increases)

3. **Asymmetry/skewness of post-test 1 distribution:**
   - Skewed distributions → uncertain sensory map
   - Could predict learning rate differences

### From Adaptation Phase:
4. **Early error variance (σ_error, trials 1-10):**
   - Reflects initial mismatch magnitude
   - Could interact with Δπ: `R = f(Δπ × σ_error_early)`

5. **Learning rate decay time constant (τ):**
   - Fit `y_t = a·exp(-t/τ) + b` to adaptation curve
   - Use τ as outcome variable (instead of just early slope)

6. **Trial-to-trial autocorrelation (ρ):**
   - Reflects state-space dynamics
   - Higher ρ → lower effective Q
   - Implementation: Compute `corr(e_t, e_{t-1})` per subject; test `ρ ~ Δπ`

### Cross-Phase Indices:
7. **Baseline precision (π_pre):**
   - Absolute starting point; Δπ relative to baseline might matter more than absolute change
   - Implementation: `R = f(π_pre, Δπ)` or `R = f(Δπ/π_pre)`

8. **Post1/Pre ratio (already using Δlogπ for this):**
   - Proportional change vs additive change
   - Current best practice: Use Δlogπ for multiplicative models, Δπ for additive

### Cognitive/Behavioral Covariates:
9. **Age, handedness, task experience:** Could modulate tuning effects
10. **Post-adaptation awareness:** Ask subjects if they noticed perturbation; test if awareness interacts with Δπ

**Recommendation:** Start with Δlogπ alone (simplest). If effects are weak, test σ_post1 or Δσ as additional predictors in exploratory analysis.

---

## Implementation Guide

### Model Structure in Code

All models follow this template:
```python
def compute_r_[model_name](r_post1, delta_logpi, **params):
    """
    Compute measurement noise R for [model_name].

    Args:
        r_post1: Baseline measurement noise (1/precision_post1)
        delta_logpi: Log-precision change (proportional)
        **params: Model-specific parameters (e.g., beta, beta_q)

    Returns:
        r: Modulated measurement noise (positive scalar)
    """
    # Model-specific computation
    r = ...
    return max(r, 1e-8)  # Safety bound
```

### Adding New Models (Cookbook):

1. **Add compute function to `src/prism_sim/model.py`:**
```python
def compute_r_mq(q_baseline: float, beta_q: float, delta_logpi: float) -> float:
    """Process noise modulation: Q = Q_base * exp(beta_q * delta_logpi)."""
    q = q_baseline * math.exp(beta_q * delta_logpi)
    return max(q, 1e-8)
```

2. **Update dispatcher in `compute_r()` function:**
```python
if model in {"MQ", "M-Q"}:
    q = compute_r_mq(Q_baseline, beta, delta_logpi)
    return q  # Return Q instead of R
```

3. **Add NumPyro model in `scripts/fit_real_data_numpyro.py`:**
```python
elif model_name in {"MQ", "M-Q"}:
    beta_q = numpyro.sample("beta_q", dist.Normal(0.0, 1.0).expand((n_groups,)))
    q_modulated = Q * jnp.exp(beta_q[group_idx] * delta_pi)
    r_measure = r_post1  # R stays fixed
    # Pass q_modulated to Kalman filter
```

4. **Modify Kalman filter to accept Q parameter:**
```python
def kalman_loglik(errors, r_measure, q_process, m, A, b):
    # ...
    p_pred = A * p * A + q_process  # Use modulated Q
    # ...
```

5. **Update priors:**
- For β_Q in M-Q: `Normal(0, 1)` (same as β_R)
- For π_vis in M3-cc: `Gamma(2, 2)` (mean=1, prior guess that visual ≈ proprioceptive precision)
- For γ in M4-joint: `Normal(0, 2)` (expect smaller effect on b than on R)

---

## Testing Protocol

### Phase 1: Fast Model Screening (Use Subset)
- **Subjects:** N=20 per group (60 total) for quick iteration
- **Chains:** 2 chains × 500 warmup × 500 samples (for speed)
- **Metric:** WAIC (faster than LOO for screening)
- **Goal:** Identify which models have lower WAIC than M0/M1; drop obviously bad models

### Phase 2: Full Model Comparison
- **Subjects:** All subjects with valid data
- **Chains:** 4 chains × 1000 warmup × 1000 samples (for convergence)
- **Metrics:** WAIC, LOO, max Rhat, min ESS
- **Diagnostics:**
  - Rhat < 1.01 (convergence)
  - ESS_bulk > 400 per parameter (effective sample size)
  - No divergences (NUTS quality)
  - LOO Pareto-k < 0.7 for >95% of subjects (no outliers)

### Phase 3: Posterior Predictive Checks
For best model(s):
1. **Early learning slope:** Compare predicted vs observed (trials 1-15)
2. **Late plateau:** Compare predicted asymptote vs observed (trials 40-50)
3. **Group ordering:** Does model preserve EC > EO+ > EO− ordering in Kalman gain?
4. **Residual patterns:** Plot `observed - predicted` vs trial, Δπ, R_post1 to check for systematic misfit

### Phase 4: Parameter Recovery Simulation
- Simulate data from best model with known parameters
- Re-fit to check bias, coverage, identifiability
- Goal: Ensure posteriors are not just fitting noise

---

## Expected Results & Interpretation

### If M1-exp wins:
- **Story:** "Proprioceptive tuning multiplicatively modulates measurement noise in visuomotor adaptation. Sharper proprioception → [higher/lower] effective sensory uncertainty."
- **Mechanism:** Sensory weighting / cue integration
- **Future work:** Independent manipulation of visual noise to test cue-combination predictions (M3-cc)

### If M-Q wins:
- **Story:** "Pre-adaptation tunes internal model volatility, not sensory noise. Proprioceptive recalibration signals environmental changeability, affecting subsequent learning rates."
- **Mechanism:** Meta-learning / volatility estimation
- **Future work:** Test with variable vs stable perturbations; change-point paradigms

### If M-hybrid wins (both β_R and β_Q nonzero):
- **Story:** "Proprioceptive tuning operates on multiple levels: recalibrating both sensory reliability (R) and internal model volatility (Q). Suggests sophisticated, multi-parameter meta-learning."
- **Mechanism:** Multi-level Bayesian adaptation
- **Future work:** Lesion studies (can block R vs Q effects separately?); neural correlates

### If all models fail (WAIC ≈ M0):
- **Story:** "Proprioceptive tuning does not reliably modulate sensorimotor adaptation in our paradigm. Possible explanations: (1) effect size too small given measurement noise, (2) tuning effects are context-specific (not captured by Δπ alone), (3) adaptation uses different sensory channels."
- **Pivot:** Exploratory analyses (test σ_post1, Δσ, subgroup analyses); or reframe as "bounds on effect size."

---

## Key References (Cite These)

### Kalman Filtering & Motor Learning:
1. **Wolpert, Ghahramani, & Jordan (1995).** "An internal model for sensorimotor integration." *Science*, 269(5232), 1880-1882.
2. **Körding & Wolpert (2004).** "Bayesian integration in sensorimotor learning." *Nature*, 427(6971), 244-247.
3. **Wei & Körding (2009).** "Relevance of error: What drives motor adaptation?" *Journal of Neurophysiology*, 101(2), 655-664.
4. **Wei & Körding (2010).** "Uncertainty of feedback and state estimation determines the speed of motor adaptation." *Frontiers in Computational Neuroscience*, 4, 11.

### Cue Combination & Sensory Integration:
5. **Ernst & Banks (2002).** "Humans integrate visual and haptic information in a statistically optimal fashion." *Nature*, 415(6870), 429-433.
6. **van Beers, Sittig, & Gon (1999).** "Integration of proprioceptive and visual position-information: An experimentally supported model." *Journal of Neurophysiology*, 81(3), 1355-1364.

### Meta-Learning & Volatility:
7. **Behrens, Woolrich, Walton, & Rushworth (2007).** "Learning the value of information in an uncertain world." *Nature Neuroscience*, 10(9), 1214-1221.
8. **Nassar, Wilson, Heasly, & Gold (2010).** "An approximately Bayesian delta-rule model explains the dynamics of belief updating in a changing environment." *Journal of Neuroscience*, 30(37), 12366-12378.
9. **Nassar, Rumsey, Wilson, Parikh, Heasly, & Gold (2012).** "Rational regulation of learning dynamics by pupil-linked arousal systems." *Nature Neuroscience*, 15(7), 1040-1046.

### Prism Adaptation & Sensory Realignment:
10. **Welch & Warren (1980).** "Immediate perceptual response to intersensory discrepancy." *Psychological Bulletin*, 88(3), 638-667.
11. **Redding & Wallace (1997).** "Adaptive spatial alignment." *Psychology Press*.
12. **Synofzik, Lindner, & Thier (2008).** "The cerebellum updates predictions about the visual consequences of one's behavior." *Current Biology*, 18(11), 814-818.

---

## Decision Criteria & Model Selection

### Quantitative Criteria:
- **ΔWAIC > 10:** Strong evidence for better model
- **ΔWAIC 4-10:** Moderate evidence
- **ΔWAIC < 4:** Weak/no evidence; prefer simpler model
- **LOO Pareto-k:** If >5% subjects have k > 0.7, model has outlier sensitivity (investigate)

### Qualitative Criteria:
1. **Mechanistic interpretability:** Can you explain β sign/magnitude in cognitive terms?
2. **Group ordering preservation:** Does model preserve EC > EO+ > EO− pattern (if it exists)?
3. **Posterior plausibility:** Are β values in reasonable range (e.g., |β| < 2 for exp models)?
4. **Predictive accuracy:** Do posterior predictive checks show good fit to early slope AND late plateau?

### Tie-Breaker (If ΔWAIC < 4):
- **Prefer simpler model** (fewer parameters)
- **Prefer theoretical grounding** (which has stronger literature support?)
- **Prefer mechanistic specificity** (which makes clearer predictions for future experiments?)

---

## Notes & Caveats

### Metric Choice (Δπ vs Δlogπ):
- **Δπ = precision_post1 - precision_pre:** Additive change; works naturally with additive links (M1)
- **Δlogπ = log(precision_post1/precision_pre):** Proportional change; works naturally with multiplicative links (M1-exp, M-Q, M-hybrid)
- **Current best practice:** Use Δlogπ for all exponential models; keeps effects proportional and avoids over-weighting high-precision subjects

### Priors:
- **β ~ Normal(0, 1):** Weakly informative; allows |β| up to ~2 (covers exp(2×0.7)≈4x scaling, which is plausible upper bound)
- **b ~ Normal(0, 30):** Covers full perturbation range (m = -12.1 diopters; b could be ±5 diopters residual)
- **Q ~ Fixed at 1e-4:** Process noise is small for short timescales; could be freed in M-Q but start with baseline from literature

### Saturation Warnings:
- **Tanh:** If posteriors show >20% saturation (|λ| > 0.95), model is over-parameterized for data range
- **Exp:** If β posterior mean > 1 and max(Δlogπ) > 1, check for runaway inflation (plot R vs Δlogπ)

### Computational Notes:
- **JAX + NumPyro:** Enable `jax_enable_x64=True` for numerical stability with small variances
- **Parallelization:** Use `chain_method='parallel'` for 4+ chains on M2 Mac (8 cores)
- **Speed:** M0-M2 take ~5-10 min per model (N=60, 4 chains × 1000 samples). M-hybrid may be 2x slower (more parameters).

---

## Summary Decision Matrix (UPDATED WITH EMPIRICAL RESULTS)

| Model       | Mechanism            | Params/Group | Empirical Result | WAIC (Δ from M0) | Status |
|-------------|----------------------|--------------|------------------|------------------|--------|
| **M0**      | None (baseline)      | 0            | ✅ Converged     | 46983 (0)        | **Baseline** |
| **M1**      | R additive           | 1 (β)        | ✅ Converged     | 44688 (-2295)    | **Strong** |
| **M1-exp**  | R multiplicative     | 1 (β)        | ✅ Converged     | 44425 (-2558)    | ✅ **WINNER** |
| **M-Q**     | Q modulation         | 1 (β_Q)      | ❌ FAILED (Rhat=1.53, ESS=7) | 57421 (+10438) | ❌ **Abandon** |
| **M-hybrid**| R + Q joint          | 2 (β_R, β_Q) | ❌ FAILED (Rhat=1.53, ESS=7) | 48160 (+1177)  | ❌ **Abandon** |
| **M2**      | R tanh (saturated)   | 1 (λ)        | ✅ Converged     | 45715 (-1268)    | Supplement |
| M3-cc       | Cue combination      | 1+ (π_vis)   | Not tested       | —                | Future |
| M4-joint    | R + plateau          | 2 (β, γ)     | Not tested       | —                | Skip |
| M1-asym     | R sign-specific      | 2 (β+, β-)   | Not tested       | —                | Skip |
| M5-gated    | R error-dependent    | 2+ (β, c)    | Not tested       | —                | Skip |

### Key Takeaways:
- **M1-exp wins:** Best WAIC, converged, theoretically sound
- **M-Q/M-hybrid catastrophic failures:** Non-identifiable with current data/priors; Q modulation empirically intractable
- **M1 is solid:** Good alternative if simplicity preferred over multiplicative interpretation
- **M2 saturated:** As predicted; no advantage over simpler models

---

## Final Recommendations (POST-EMPIRICAL RESULTS)

### For Main Paper:
1. **Report M0, M1, M1-exp only** (drop M-Q/M-hybrid due to convergence failure)
2. **WAIC table:** Show clear progression M0 → M1 → M1-exp with ΔWAIC values
3. **Parameter estimates:** Report β for each group (EC, EO+, EO-) with 95% credible intervals
4. **Interpretation:** Address counter-intuitive positive β:
   - "Higher Δlogπ → higher R → lower Kalman gain → slower adaptation"
   - Discuss source estimation / sensory reweighting mechanisms
   - Acknowledge multicollinearity concern with R_post1
5. **Posterior predictive checks:** Show M1-exp captures early learning slopes and late plateaus
6. **Group differences:** Test if β truly differs across EC/EO+/EO- or if fixed β is sufficient

### For Supplement:
- **M2 saturation analysis:** Show λ saturates at -0.64 to -0.73; methodological lesson about bounded models with small data ranges
- **M-Q/M-hybrid failure documentation:** Explain non-convergence as cautionary tale; discuss identifiability issues when Q and R both modulated
- **Sensitivity analyses:**
  - Δπ vs Δlogπ metric (already done; logpi superior for exp models)
  - Fixed vs group-specific β (test parsimony)
  - Prior sensitivity: β ~ Normal(0, 0.5) vs Normal(0, 1) for M1-exp
- **Diagnostic plots:** Trace plots, pair plots, R² for M1-exp

### For Discussion / Future Work:
- **Why did Q modulation fail?**
  - Theoretical: Q and R have similar effects on Kalman gain; non-identifiable without independent manipulation
  - Empirical: Prior allowed Q to explode/collapse; tighter priors (Normal(0, 0.05)) might help but unlikely to change conclusion
  - Recommendation: Future studies should manipulate volatility directly (variable vs stable perturbations)
- **Alternative mechanisms to test:**
  - M3-cc (cue combination) with independent visual noise estimates
  - State uncertainty modulation (initial P0) instead of Q
  - Non-linear Δπ effects (quadratic term)
- **Neural correlates:** fMRI/EEG to test if R modulation corresponds to cerebellar or parietal activity changes

### Immediate Next Steps:
1. ✅ Run posterior predictive checks for M1-exp
2. ✅ Plot β posteriors with group comparisons
3. ✅ Compute R² or similar fit metric for M0/M1/M1-exp
4. ✅ Test fixed-β model: does β really differ across groups?
5. ⚠️ Address counter-intuitive effect: why does sharper proprioception slow adaptation?

---

## 📊 POST-FITTING ANALYSIS & VISUALIZATION PLAN

**Status:** In progress (focusing on M0, M1, M2, M1-exp; M-Q/M-hybrid set aside)

### Phase 1: Model Comparison & Selection ✅ IN PROGRESS

#### 1.1 WAIC/LOO Comparison
- **Plot:** Bar chart showing ΔWAIC relative to M0 with SE error bars
- **Output:** `figures/model_comparison_waic.png`
- **Key insight:** Quantify M1-exp superiority (ΔWAIC = -2558 from M0)

#### 1.2 Information Criteria Table
- **Table:** Model | WAIC | ΔWAIC | SE | LOO | p_waic | LOO warnings
- **Output:** `tables/model_comparison_metrics.csv`
- **Goal:** Show evidence ratios and model weights

#### 1.3 Convergence Diagnostics Summary
- **Table:** Model | max_rhat | min_ess_bulk | min_ess_tail | converged?
- **Output:** `tables/convergence_diagnostics.csv`
- **Goal:** Validate inference quality for all models

---

### Phase 2: Parameter Estimates & Interpretation

#### 2.1 β Posterior Distributions
- **Plot:** Forest plot with 95% credible intervals
  - Y-axis: Groups (EC, EO+, EO-)
  - X-axis: β value
  - Separate facets for M1 vs M1-exp
- **Output:** `figures/beta_posteriors_by_group.png`
- **Questions:**
  - Are all β > 0? (Pr(β > 0) > 95%?)
  - Do CIs overlap across groups?
  - M1 vs M1-exp: tighter estimates?

#### 2.2 Effect Size Interpretation
- **Compute:** For typical Δlogπ (e.g., 0.2 for EC), translate β → R change → Kalman gain change
- **Table:** Group | β_median | R_change (%) | K_change (%) | Learning_rate_impact
- **Output:** `tables/effect_sizes.csv`
- **Goal:** Make β interpretable in terms of adaptation speed

#### 2.3 Plateau (b) Estimates
- **Plot:** Bar plot of b by group for each model
- **Output:** `figures/plateau_by_group.png`
- **Question:** Does b vary systematically with group or Δπ?

---

### Phase 3: Posterior Predictive Checks (PPC) 🔥 CRITICAL

#### 3.1 Adaptation Curves: Predicted vs Observed
- **Plot:** 3-panel figure (EC | EO+ | EO-)
  - X-axis: Trial (1-100)
  - Y-axis: Pointing error (°)
  - Lines:
    - Observed mean ± SE (black solid)
    - M0 prediction (gray dashed, baseline)
    - M1 prediction (blue dotted)
    - M1-exp prediction (red solid, 95% PI shaded)
    - M2 prediction (purple dotted)
- **Output:** `figures/ppc_adaptation_curves_by_group.png`
- **Questions:**
  - Does M1-exp capture early steep descent?
  - Plateau accuracy?
  - Where do M1 vs M1-exp diverge most?

#### 3.2 Subject-Level Fits (Sample)
- **Plot:** Random sample of 9 subjects (3 per group)
  - Individual trajectories with model predictions overlaid
- **Output:** `figures/ppc_subject_sample.png`
- **Goal:** Show model captures individual variability

#### 3.3 Residual Analysis
- **Plot:** 4-panel diagnostic
  - Panel A: Residuals vs Trial (check time trends)
  - Panel B: Residuals vs Δlogπ (check if Δπ effect captured)
  - Panel C: Residuals vs R_post1 (check multicollinearity)
  - Panel D: Q-Q plot (check normality)
- **Output:** `figures/residuals_diagnostics.png`
- **Goal:** Validate no systematic misfit

---

### Phase 4: Mechanism Validation

#### 4.1 Δπ → R Relationship
- **Plot:** Scatter plot with model overlays
  - X-axis: Δlogπ (subject-level)
  - Y-axis: Predicted R
  - Color by group (EC, EO+, EO-)
  - Lines: M0 (horizontal), M1 (linear), M1-exp (exponential), M2 (tanh)
- **Output:** `figures/deltalogpi_vs_R_models.png`
- **Questions:**
  - How different are functional forms?
  - Does M1-exp curvature matter at observed Δπ range?
  - Is M2 saturated (flat)?

#### 4.2 Kalman Gain vs Δπ
- **Plot:** Predicted gain K = P/(P + R) vs Δlogπ
  - Separate lines per model
  - Overlay empirical "early learning slope" proxy
- **Output:** `figures/kalman_gain_vs_deltalogpi.png`
- **Goal:** Show β translates to meaningful gain differences

#### 4.3 Early Learning Slope vs Δπ (Empirical Check)
- **Compute:** Slope = (mean error trials 1-15) - (mean error trials 10-25) per subject
- **Plot:** Scatter Δlogπ vs empirical slope, color by group
  - Overlay model predictions
- **Output:** `figures/empirical_slope_vs_deltalogpi.png`
- **Question:** Do high Δπ subjects actually learn slower? (validates positive β)

---

### Phase 5: Multicollinearity & Confounding Checks

#### 5.1 Parameter Correlation Matrix
- **Plot:** Pair plot from posterior samples (M1-exp only)
  - Variables: β_EC, β_EO+, β_EO-, b_EC, b_EO+, b_EO-
  - Check: β correlated with b? (independence)
- **Output:** `figures/parameter_correlations.png`
- **Goal:** Check posterior dependencies

#### 5.2 β vs R_post1 Correlation
- **Plot:** Scatter of median β_posterior vs R_post1 per subject
- **Stat:** Compute Pearson r and p-value
- **Output:** `figures/beta_vs_rpost1_correlation.png`
- **Goal:** Rule out that β just captures baseline R differences (spurious)

---

### Phase 6: Group-Level Comparisons

#### 6.1 Test Fixed β vs Group-Specific β
- **Refit:** M1-exp with single β shared across groups
- **Compare:** WAIC_fixed vs WAIC_grouped
  - If ΔWAIC < 4: groups don't differ; prefer simpler model
- **Output:** `tables/fixed_vs_grouped_beta.csv`

#### 6.2 Pairwise β Differences
- **Compute:** Posterior distributions of β_EC - β_EO+, β_EC - β_EO-, β_EO+ - β_EO-
- **Plot:** Violin plots with 95% CI
- **Output:** `figures/pairwise_beta_differences.png`
- **Question:** Does visual feedback during pre-adaptation matter? (EO+ vs EO-)

---

### Phase 7: Model-Specific Diagnostics

#### 7.1 M2 Saturation Analysis
- **Plot:** Histogram of λ posteriors per group
  - Overlay ±0.95 saturation boundaries
  - Annotate % samples with |λ| > 0.9
- **Output:** `figures/m2_lambda_saturation.png`
- **Goal:** Confirm M2 over-parameterized

#### 7.2 Convergence Diagnostics (Trace Plots)
- **Plot:** Trace plots for β, b, λ (all models, select chains)
- **Output:** `figures/trace_plots_all_models.png`
- **Goal:** Visual confirmation of mixing

---

## 📐 Priority Implementation Order

### **Immediate (Week 1):**
1. ✅ Phase 1.1: WAIC comparison plot
2. ✅ Phase 2.1: β posterior distributions
3. ✅ Phase 3.1: Adaptation curves PPC (group-level)
4. Phase 4.1: Δπ → R relationship

### **High Priority (Week 2):**
5. Phase 3.3: Residual diagnostics
6. Phase 4.3: Early slope vs Δπ (empirical validation)
7. Phase 6.1: Fixed β comparison
8. Phase 2.2: Effect size table

### **Medium Priority (Week 3):**
9. Phase 5.2: β vs R_post1 correlation check
10. Phase 3.2: Subject-level fits sample
11. Phase 7.1: M2 saturation analysis

### **Lower Priority (Supplement):**
12. Phase 6.2: Pairwise β differences
13. Phase 4.2: Kalman gain plot
14. Phase 7.2: Trace plots

---

**END OF REPORT**

*This document should be updated after each round of model fitting with results, convergence diagnostics, and revised recommendations.*
