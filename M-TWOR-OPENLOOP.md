# R_obs Model Variants: Empirical Grounding and Decomposition

This document describes alternative parameterizations of observation noise (R_obs) tested to understand what cognitive/sensory factors contribute to trial-to-trial variability during visuomotor adaptation.

---

## Summary of Models Tested

| Model | R_obs Parameterization | WAIC | Converged? | Result |
|-------|------------------------|------|------------|--------|
| **M-TWOR** | R_obs_base (free) × exp(β_obs × Δπ) | 35,410 | ✓ | **Best fit** |
| **M-TWOR-SENSORY** | (openloop + visual) + R_cognitive × exp(β_obs × Δπ) | 35,488 | ✓ | Good fit, interpretable |
| M-TWOR-OPENLOOP | openloop_var × exp(β_obs × Δπ) | 39,848 | ✗ | Failed |

---

# Part 1: M-TWOR-OPENLOOP (Failed)

## Motivation

We hypothesized that R_obs represents motor execution noise, which could be measured directly via **openloop reaching** (reaching without visual feedback).

### Formula
```
R_obs = openloop_var_post1 × exp(β_obs × Δlog π)
```

## Result: Model Failed

| Metric | Value | Criterion |
|--------|-------|-----------|
| WAIC | 39,848 | +4,438 worse than M-TWOR |
| R̂ | 1.53 | Should be < 1.01 |
| ESS | 7 | Should be > 400 |

### Why It Failed

**Scale mismatch:**
| Measure | Value |
|---------|-------|
| Openloop variance (empirical) | ~1.1 - 1.5 |
| R_obs_base (estimated in M-TWOR) | ~5.3 - 5.8 |

The estimated R_obs is **~4x larger** than openloop motor variance, indicating R_obs captures much more than just motor execution noise.

### Conclusion

R_obs ≠ pure motor execution noise. It includes additional sources of variability.

---

# Part 2: M-TWOR-SENSORY (Success)

## Motivation

Building on the M-TWOR-OPENLOOP failure, we decomposed R_obs into:
1. **Sensory noise** (fixed, empirical): motor + visual
2. **Cognitive noise** (estimated, modulated): attention, strategy, unmeasured factors

### Formula
```
R_obs = (openloop_var + visual_var) + R_cognitive × exp(β_obs × Δlog π)
         \___________ ___________/          \_____________ ______________/
                     V                                     V
            Sensory (fixed)                    Cognitive (modulated by Δπ)
```

### Cognitive Rationale

| Component | Source | Measured? | Modulated by Δπ? |
|-----------|--------|-----------|------------------|
| **R_motor** | Motor execution noise | ✓ Openloop reaching | No (fixed) |
| **R_visual** | Visual encoding noise | ✓ Visual localization | No (fixed) |
| **R_cognitive** | Attention, strategy | Estimated | **Yes** |

**Key insight**: Sensory noise should NOT change with proprioceptive precision. Only cognitive factors might be modulated.

## Result: Model Succeeded

| Metric | M-TWOR | M-TWOR-SENSORY |
|--------|--------|----------------|
| WAIC | 35,410 | 35,488 |
| ΔWAIC | 0 | +78 (small) |
| R̂ | 1.00 | 1.00 |
| ESS | 3,571 | 2,721 |

**Conclusion**: M-TWOR-SENSORY converged well and fits nearly as well as M-TWOR.

---

## Parameter Estimates

### β_state (Learning Mechanism) — Robust Across Models

| Group | M-TWOR | M-TWOR-SENSORY | Interpretation |
|-------|--------|----------------|----------------|
| EC | +1.098 | +1.047 | Strongest source-estimation |
| EO+ | +0.674 | +0.666 | Moderate |
| EO- | +0.488 | +0.480 | Weakest but still positive |

**Key finding**: β_state > 0 is **robust** regardless of R_obs parameterization. Source-estimation is supported.

### R_cognitive (Cognitive Noise Component)

| Group | R_cognitive | Sensory (motor+visual) | Total R_obs | % Cognitive |
|-------|-------------|------------------------|-------------|-------------|
| EC | 4.30 | 1.55 | 5.85 | **73%** |
| EO+ | 3.79 | 2.00 | 5.79 | **65%** |
| EO- | 3.78 | 2.08 | 5.86 | **65%** |

**Key finding**: ~65-73% of observation noise is **cognitive**, only ~27-35% is sensory.

### β_obs (Now Specifically Cognitive Modulation)

| Group | M-TWOR | M-TWOR-SENSORY | Interpretation |
|-------|--------|----------------|----------------|
| EC | -1.18 | **-1.65** | Strong: Δπ ↑ → cognitive noise ↓ |
| EO+ | -0.41 | -0.49 | Moderate effect |
| EO- | +0.09 | +0.06 | No effect |

**Key finding**: In M-TWOR-SENSORY, β_obs specifically captures how proprioceptive tuning modulates **cognitive noise**:
- EC group: Sharper proprioception → more focused attention → less cognitive noise
- EO- group: No cognitive modulation

---

## Theoretical Interpretation

### The Decomposition Story

```
During visuomotor adaptation:

Total Observation Noise (R_obs ≈ 5.8)
├── Sensory Noise (~1.8, fixed)
│   ├── Motor execution (openloop_var ≈ 1.3)
│   └── Visual encoding (visual_var ≈ 0.5)
│
└── Cognitive Noise (~4.0, modulated by Δπ)
    ├── Attention fluctuations
    ├── Strategy variability
    └── Other unmeasured factors
```

### Why EC Shows Strongest β_obs Effect

1. **Eyes-closed condition** requires sustained proprioceptive attention
2. When proprioceptive precision increases (Δπ ↑):
   - Attention becomes more stable
   - Cognitive noise decreases (β_obs < 0)
3. **EO- condition** has visual system active but uninformative:
   - Attentional conflict between vision and proprioception
   - Proprioceptive tuning doesn't stabilize attention
   - No β_obs effect

---

## Summary for Paper

### Methods Statement
> "We tested whether observation noise (R_obs) corresponds to motor execution variability by constraining it to empirically measured openloop reaching variance (M-TWOR-OPENLOOP). This model failed to converge (R̂ = 1.53), indicating that R_obs captures additional sources of variability beyond motor execution."

### Results Statement
> "Decomposing observation noise into sensory and cognitive components (M-TWOR-SENSORY) revealed that approximately 70% of trial-to-trial variability reflects cognitive factors (R_cognitive ≈ 3.8-4.3), while only 30% reflects sensory noise (motor + visual ≈ 1.5-2.1). Proprioceptive tuning specifically modulated the cognitive component (β_obs), with the strongest effect in the eyes-closed group (β_obs = -1.65), suggesting that enhanced proprioceptive precision stabilizes attention during visuomotor adaptation. Critically, the learning mechanism parameter (β_state) remained robust across model specifications (M-TWOR: 0.49-1.10; M-TWOR-SENSORY: 0.48-1.05), confirming that the source-estimation effect is not an artifact of R_obs parameterization."

---

## Data Summary

### Empirical Variance Measures (Post1)

| Group | N | Openloop Var | Visual Var | Sensory Total |
|-------|---|--------------|------------|---------------|
| EC | 25 | 1.14 ± 0.60 | 0.41 ± 0.29 | 1.55 |
| EO+ | 24 | 1.41 ± 1.54 | 0.59 ± 0.72 | 2.00 |
| EO- | 24 | 1.49 ± 0.86 | 0.59 ± 0.57 | 2.08 |

### Full Model Comparison

| Model | WAIC | ΔWAIC | R̂ | ESS | Status |
|-------|------|-------|-----|-----|--------|
| M-obs-fixed | 35,470 | +60 | 1.00 | 4,587 | Baseline |
| M-obs | 35,446 | +36 | 1.00 | 5,527 | OK |
| **M-twoR** | **35,410** | **0** | **1.00** | **3,571** | **Best** |
| M-twoR-openloop | 39,848 | +4,438 | 1.53 | 7 | Failed |
| M-twoR-sensory | 35,488 | +78 | 1.00 | 2,721 | Good |

---

## Running the Models

### Extract Data
```bash
poetry run python scripts/data_processing/extract_prepost.py
```

### Fit All Models
```bash
poetry run python scripts/modeling/fit_real_data_m_obs.py \
    --plateau-group-specific \
    --models "M-twoR,M-twoR-openloop,M-twoR-sensory"
```

### Fit Only Comparison Models
```bash
poetry run python scripts/modeling/fit_real_data_m_obs.py \
    --plateau-group-specific \
    --models "M-twoR,M-twoR-sensory"
```

---

## References

- Original M-TWOR model: see `paper.txt`, `RESULTS.md`
- Data extraction: `scripts/data_processing/extract_prepost.py`
- Model fitting: `scripts/modeling/fit_real_data_m_obs.py`

---

## Changelog

- **2024-01-10**: Added M-TWOR-OPENLOOP (failed) and M-TWOR-SENSORY (success)
- **2024-01-10**: Documented decomposition results showing ~70% cognitive, ~30% sensory noise
