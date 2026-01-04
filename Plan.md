# Precision Transfer in Prism Adaptation  
## Research Plan: Scope, Simulation, and Model Fitting

---

## 1. Scope and Goal of This Research

### 1.1 Core Research Question

This project investigates whether **sensory precision established prior to perturbation exposure**—specifically, proprioceptive precision—**transfers to a novel perturbation context** and regulates how sensorimotor errors are utilized during learning.

The central questions are:

1. Does proprioceptive reliability at the onset of prism exposure influence subsequent error correction?
2. If so, *how* is this influence implemented computationally—via a linear modulation of error uncertainty, or via a bounded, saturating mechanism?

---

### 1.2 Experimental Context and Design Implications

The pre-adaptation phase manipulates visual availability during reaching (EO+, EO-, EC) to shape proprioceptive precision profiles before prism exposure. This yields a participant-specific tuning signal
\[
\Delta\pi_i = \log \pi^{\mathrm{post1}}_{P,i} - \log \pi^{\mathrm{pre}}_{P,i},
\]
which can be positive (sharpened precision) or negative (blunted precision). The key mechanistic claim is that these pre-exposure precision changes transfer into prism adaptation via their effect on the effective measurement noise \(R_i\).

Because \(\Delta\pi_i\) is signed, the sign of the modulation parameter (especially \(\lambda\) in M2) is theoretically meaningful:
- **\(\lambda > 0\)**: \(\Delta\pi_i>0\) reduces \(R_i\) (greater error utilization); \(\Delta\pi_i<0\) increases \(R_i\) (more conservative learning).
- **\(\lambda < 0\)**: the mapping reverses, implying that increased precision reduces error utilization (an attribution-dominant regime).

This sign directly shapes predicted group differences. If the tuning manipulation reliably shifts \(\Delta\pi\) by group, the expected ordering of early adaptation rates should flip under \(\lambda>0\) vs \(\lambda<0\). Distinguishing these regimes is central to the interpretive payoff of the experiment.

#### Competing \(\lambda\) regimes (reliability vs source-estimation)

- **Reliability-based route (\(\lambda>0\))**: sharper proprioception (positive \(\Delta\pi\)) lowers \(R_i\), increasing Kalman gain and speeding early adaptation. Groups with larger positive \(\Delta\pi\) should sit highest in early learning; negative \(\Delta\pi\) groups should be slowest.
- **Source-estimation route (\(\lambda<0\))**: sharper proprioception increases the tendency to attribute errors to external causes, effectively inflating \(R_i\) and dampening gains. The group ordering in early adaptation should invert relative to the \(\lambda>0\) regime.

Planned fits (Section 4) explicitly estimate group-specific \(\lambda_g\) to adjudicate which route the data support. Early-trial group ordering serves as an interpretable cross-check against these two sign-specific predictions (see note3 schematic).

---

### 1.3 Modeling Perspective

Prism adaptation is framed as a **state-space learning process** in which endpoint error serves as a measurement signal. Learning dynamics are governed by the Kalman gain, which depends critically on the **measurement noise term \(R\)**.

The key hypothesis is that \(R\) is **not fixed across individuals**, but instead depends on proprioceptive state measured prior to adaptation.

We consider three competing parameterizations of \(R_i\):

- **M0 (baseline)**  
  \( R_i = R^{\mathrm{post1}}_{P,i} \)

- **M1 (linear additive modulation)**  
  \( R_i = R^{\mathrm{post1}}_{P,i} + \beta\,\Delta\pi_i \)

- **M2 (bounded multiplicative modulation)**  
  \( R_i = R^{\mathrm{post1}}_{P,i}\bigl(1 - \lambda\,\tanh(\Delta\pi_i)\bigr), \quad \lambda \in [-1,1] \)

where:
- \( \pi = 1/\sigma^2 \) denotes proprioceptive precision,
- \( \Delta\pi_i = \pi^{\mathrm{post1}}_i - \pi^{\mathrm{pre}}_i \).

---

### 1.4 Overall Strategy

- **Primary goal**: parameter recovery and interpretability of the \(R\)-mapping parameters (\(\lambda\), \(\beta\)).
- **Secondary goal**: identify and control for confounds, especially non-zero error plateaus.
- **Model comparison** is conducted, but selection power is not the primary optimization target.

Accordingly, the workflow is divided into:
1. **Pre-model empirical checks** (construct separability and phenomenology),
2. **Simulation** (recovery- and confound-focused),
3. **Real data fitting** (mechanistic inference with robustness analyses).

---

## 2. Pre-Model Checks: Construct Separability and Mapping

These analyses serve to (i) verify that predictors capture distinct aspects of learner state, and (ii) establish a phenomenological association between proprioceptive reliability at onset and subsequent error-correction behavior.  
They do **not** adjudicate between mechanistic models.

---

### 2.1 Check 1: Independence of Onset Reliability and Tuning Direction

To avoid conceptual redundancy (i.e., encoding the same signal twice), we assess whether post-test 1 proprioceptive reliability and tuning direction (\(\Delta\pi\)) are strongly collinear at the participant level.

- Predictor 1: \( \log \pi_{P,i}^{\mathrm{post1}} \) (onset reliability)
- Predictor 2: \( \Delta\pi_i \) (tuning direction)

We estimate their association using a **Bayesian correlation or regression model**, reporting the posterior for the slope (or correlation) with credible intervals.

**Interpretation**:
- A weak association supports treating onset reliability and tuning direction as dissociable inputs to the mechanistic models.
- Strong collinearity signals an identifiability risk for M1/M2 and motivates caution or rescaling.

Additional group-level check:
- Verify that \(\Delta\pi\) distributions differ across EO+, EO-, EC in the intended direction and that overlap is not so large as to obscure sign-dependent effects.

---

### 2.2 Check 2: Phenomenological Precision–Behavior Mapping

To visualize whether proprioceptive reliability at adaptation onset covaries with error utilization, we compute a **behavioral proxy** of early learning.

At least one of the following is reported (the other may serve as a robustness check):

#### Option A: Exponential Early-Phase Summary
Fit an exponential-with-asymptote to early exposure trials:
\[
e_{i,t} = a_i \exp(-t/\tau_i) + b_i + \epsilon_{i,t},
\]
and define the early learning rate proxy:
\[
\alpha_i = 1/\tau_i.
\]

#### Option B: Error-to-Update Sensitivity
Define \( \Delta y_{i,t+1} = y_{i,t+1} - y_{i,t} \) and estimate:
\[
\Delta y_{i,t+1} = c_i + \gamma_i e_{i,t} + \eta_{i,t}.
\]

The proxy (\(\alpha_i\) or \(\gamma_i\)) is then related to \( \log \pi_{P,i} \) using Bayesian regression, optionally including group and interaction terms.

These mappings are reported as supportive evidence (often in Supplementary Materials); mechanistic inference relies on the models below.

Optional targeted check:
- Stratify the proxy mapping by the sign of \(\Delta\pi\) to visualize whether learning changes are monotonic with \(\Delta\pi\) and whether any reversal suggests \(\lambda<0\).

---

## 3. Simulation Plan (Recovery and Confound-Oriented)

### 3.1 Purpose of Simulation

Simulation is used to answer:

1. Can the true \(R\)-mapping parameters (\(\lambda\), \(\beta\)) be recovered?
2. Under what conditions does recovery degrade?
3. Can non-zero error plateaus masquerade as precision-based modulation?

Model selection accuracy per se is not the focus.

---

### 3.2 Generative Model

#### State-Space Structure
Scalar state-space model:

- **State equation**
  \[
  x_{t+1} = A x_t + w_t,\quad w_t \sim \mathcal{N}(0,Q)
  \]

- **Measurement equation**
  \[
  e_t = m - x_t + b + v_t,\quad v_t \sim \mathcal{N}(0,R_i)
  \]

Conventions:
- Error \(e_t =\) hand – target (right positive, left negative).
- Prism perturbation: leftward 12.1 cm, hence \( m = -12.1 \).
- \(x_t\): internal compensation state.
- \(b\): late-stage error plateau (optional).

---

### 3.3 Fixed Parameters

Held constant to preserve identifiability:

- State dimension: scalar, \(H=1\)
- \(A = 1\) (or 0.95 in sensitivity checks)
- \(Q\): small constant (e.g., \(10^{-4}\)–\(10^{-3}\))
- Trial length: 100
- Baseline scale of \(R^{\mathrm{post1}}_{P,i}\): matched to empirical proprioceptive variance

---

### 3.4 Swept Parameters

1. **Mechanism strength**
   - \( \lambda \in \{-0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8\} \)  
   - or \(\beta\) at small / medium / large magnitudes

2. **Scale of tuning**
   - SD of \(\Delta\pi\): small (linear), medium, large (saturated \(\tanh\))

3. **Collinearity stress**
   - \( \rho(R^{\mathrm{post1}}_{P}, \Delta\pi) \in \{0, 0.3, 0.6\} \)

4. **Plateau confound**
   - \( b \in \{0, b_0\} \), with \( b_0 \approx 10\%\text{–}20\% \, |m| \)
   - Fit models with and without \(b\)

5. **Group structure (design realism)**
   - Generate \(\Delta\pi\) with group-specific means/SDs reflecting EO+, EO-, EC.
   - Evaluate whether sign recovery depends on group separation or overlap.

---

### 3.5 Evaluation Metrics

- Bias and RMSE of recovered \(\lambda\) / \(\beta\)
- 95% interval coverage
- Boundary saturation rate (posterior mass at \(\pm1\))
- Spurious recovery under plateau misspecification
- **Sign recovery**: posterior probability \(\Pr(\lambda>0)\) or \(\Pr(\lambda<0)\) under known ground truth.
- **Group-order predictions**: whether simulated group differences in early learning match the ground-truth sign regime.

---

### 3.6 Robust Simulation Checklist (Confidence Building)

A robust simulation suite should include:
1. **Parameter recovery** under realistic group-wise \(\Delta\pi\) distributions and sample sizes.
2. **Sign identifiability** of \(\lambda\) when \(\Delta\pi\) spans both positive and negative values.
3. **Model misspecification tests**, including:
   - fitting without plateau when data include \(b\),
   - fitting with plateau when data have \(b=0\),
   - varying \(A\) and \(Q\) with regularization.
4. **Noise sensitivity**, exploring broader ranges of \(R^{\mathrm{post1}}_{P}\) dispersion and trial counts.
5. **Proxy coherence**, verifying that early-phase proxies (slope, \(\tau\), \(\gamma\)) track the mechanistic effect direction.
6. **Posterior predictive checks** that reproduce early dynamics and late plateaus for each model.

---

## 4. Real Data Fitting Plan

### 4.1 Data Preparation

1. Define endpoint error:
   \[
   e_t = \text{hand endpoint} - \text{target position}
   \]
2. Set perturbation constant: \( m = -12.1 \) cm.
3. Compute proprioceptive precision:
   - \( \pi = 1/\sigma^2 \)
   - \( \Delta\pi = \pi^{\mathrm{post1}} - \pi^{\mathrm{pre}} \)
4. Re-run independence check (Section 2.1).

---

### 4.2 Mechanistic Model

State update:
\[
x_{i,t} = x_{i,t-1} + K_{i,t} e_{i,t},
\qquad
K_{i,t} = \frac{P^-_{i,t}}{P^-_{i,t}+R_i},
\quad
P^-_{i,t} = P_{i,t-1} + Q.
\]

The models differ only in the specification of \(R_i\) (M0–M2).

---

### 4.3 Primary Analysis

- Fix \(A,Q\) across models.
- Fit M0–M2 using Bayesian inference with weakly informative priors.
- Include plateau term \(b\) if late errors do not converge to zero.
- Assess fit via posterior predictive checks.
- Compare models using LOO or WAIC.

---

### 4.4 Robustness Analysis

- Allow \(A\) (and optionally \(Q\)) to vary under strong regularization.
- Refit M0–M2.
- Confirm stability of \(\lambda\) estimates and model ranking.

---

## 5. Visualization and Reporting

- Early-phase learning summaries (zoomed trajectories).
- Difference plots to localize effects across trials.
- Individual-level relationships between \(\Delta\pi\) and learning proxies.
- PPC comparisons with and without plateau terms.
- Uncertainty bands shown sparingly or in Supplementary Materials.

---

## Outcome

This plan enables a principled test of whether **precision transfer modulates error utilization** while:
- preserving parameter identifiability,
- guarding against plateau confounds,
- and aligning empirical checks with mechanistic inference.

The plan is now ready to be translated directly into simulation and fitting code.

---

## 6. Current Status and Next Actions

### 6.1 Completed

- Group-structured simulation based on empirical \(\Delta\pi\) distributions.
- Adaptation error calibration (early mean for \(m\); late mean for \(b\)).
- Recovery fitting pipeline and diagnostic plots.
- Group-level early-slope plots and improved plateau/collinearity diagnostics.

### 6.2 Next Actions (Simulation)

1. **Group-specific modulation recovery**
   - Fit simulations with group-specific \(\lambda_g\) or \(\beta_g\).
   - Evaluate identifiability and bias per group.
2. **Sensitivity to calibration and noise**
   - Sweep \(Q\) and \(R\) dispersion to match empirical variability.
   - Test robustness of recovery to plateau misspecification.
3. **Power and separation**
   - Quantify how much group separation in \(\Delta\pi\) is needed for reliable sign recovery.

### 6.3 Next Actions (Real Data)

1. **Pre-model checks**
   - Independence of \(R^{post1}_P\) and \(\Delta\pi\).
   - Precision–behavior mapping using early-learning proxies.
2. **Mechanistic fitting**
   - Fit M0–M2 to adaptation-phase error trials.
   - Compare models with LOO/WAIC and posterior predictive checks.
3. **Group effects**
   - Decide whether to model group-specific modulation parameters or group offsets.
   - Report group differences in fitted parameters and predictive trajectories.
