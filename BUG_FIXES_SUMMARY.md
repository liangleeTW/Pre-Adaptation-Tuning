# Critical Bug Fixes in NumPyro Fitting Script

## Date: 2026-01-02

## Summary
Fixed **THREE CRITICAL BUGS** in `scripts/fit_real_data_numpyro.py` that were causing:
1. Exploding log-likelihoods (17,500× too large)
2. Invalid WAIC/LOO values (millions instead of thousands)
3. Uninterpretable M2 parameters (reporting unconstrained values)

---

## Bug 1: Wrong Kalman Gain Sign (CATASTROPHIC)

### Problem
**Location:** Line 85
**Impact:** Complete invalidation of all results

The Kalman gain had the wrong sign:
```python
# WRONG (before):
k = p_pred / s
```

This caused the state to update in the OPPOSITE direction, leading to:
- State divergence (x → 132 instead of converging to ~12)
- Massive innovations (v → 138+)
- **Log-likelihoods 17,500× too large** (-67,899 per trial instead of -3.88)
- WAIC/LOO in millions instead of thousands

### Fix
```python
# CORRECT (after):
k = -p_pred / s  # Negative because observation matrix h = -1
x_new = x_pred + k * v
p_new = (1.0 + k) * p_pred  # Clearer form since k is negative
```

### Verification
- Before fix: LL = -6,789,866 for 100 trials
- After fix: LL = -388 for 100 trials
- **Improvement: 17,500× reduction**
- State now converges properly to ~-13.9 (expected ~-12.1)

---

## Bug 2: Wrong λ Parameter Reporting (M2 Uninterpretable)

### Problem
**Location:** Lines 229-244
**Impact:** M2 results completely uninterpretable

The code was reporting the RAW unconstrained parameter `lam` instead of the actual model parameter `tanh(lam)`:

```python
# WRONG (before):
lam_med = np.median(np.array(post["lam"]), axis=0)
```

This caused reported values like λ = -5.19, which is meaningless since the model uses `tanh(λ)` constrained to [-1, 1].

### Fix
```python
# CORRECT (after):
lam_raw = np.array(post["lam"])
lam_transformed = np.tanh(lam_raw)  # Apply tanh transformation
lam_med = np.median(lam_transformed, axis=0)

# Also add saturation indicators
summary[f"lam_saturated_{group}"] = (np.abs(lam_transformed) > 0.95).mean()
```

### Impact
- Now reports actual model parameters (constrained to [-1, 1])
- Adds saturation indicators to identify boundary issues
- M2 results now interpretable

---

## Bug 3: Missing Convergence Diagnostics

### Problem
No Rhat or ESS diagnostics in output, making it impossible to assess MCMC quality.

### Fix
Added computation and reporting of:
- `max_rhat`: Maximum R-hat across all parameters (should be < 1.01)
- `min_ess_bulk`: Minimum bulk ESS (should be > 400)
- `min_ess_tail`: Minimum tail ESS (should be > 400)

Warnings are printed during fitting if diagnostics are concerning.

---

## Results Comparison

### Before Fixes (INVALID):
| Metric | M0 | M1 | M2 |
|--------|-------|-------|-------|
| WAIC | 12,897,888 | 11,358,989 | 11,975,529 |
| LOO | 12,174,990 | 10,020,943 | 10,664,494 |
| p_waic | 383,601 | 695,855 | 684,773 |

**Problems:**
- Values 1000× too large
- p_waic > n_subjects (impossible!)
- All model comparison invalid

### After Fixes (Quick Test, 5 subjects):
| Metric | Value |
|--------|-------|
| WAIC | 3,097 |
| LOO | 3,097 |
| p_waic | 0.48 |
| max_rhat | 1.02 |
| min_ess_bulk | 120 |

**Results:**
- ✅ Values in reasonable range
- ✅ p_waic << n_subjects
- ✅ Convergence diagnostics included
- ⚠️ ESS low (expected for quick test with 200 samples)

---

## Action Required

### Immediate:
1. ✅ **INVALIDATE all previous results** - they are mathematically incorrect
2. ⚠️ **Re-run all fits** with corrected code
3. ⚠️ **Use more samples** for final results (draws=2000, tune=2000, chains=4)

### Recommended Run:
```bash
poetry run python scripts/fit_real_data_numpyro.py \
  --models M0,M1,M2 \
  --draws 2000 \
  --tune 2000 \
  --chains 4 \
  --target-accept 0.90 \
  --out-path data/derived/real_fit_CORRECTED.csv
```

Expected runtime: ~30-60 minutes for full dataset (69 subjects)

---

## Technical Details

### Kalman Filter Correction
The observation model is: `e_t = m - x_t + b + noise`

With observation matrix h = -1:
- Observation: `y_pred = h * x_pred + c = -x_pred + (m + b)`
- Innovation covariance: `s = h² * p_pred + r = p_pred + r`
- Kalman gain: `k = p_pred * h / s = -p_pred / s` (NEGATIVE!)
- State update: `x_new = x_pred + k * v`
- Covariance update: `p_new = (1 - k*h) * p_pred = (1 + k) * p_pred`

### State Interpretation
For prism perturbation m = -12.1 (leftward shift):
- State x converges to approximately -12 to -14
- Negative state = compensation in same direction as perturbation
- This allows error to approach plateau: `e ≈ 0 + b`

---

## Files Modified
- `scripts/fit_real_data_numpyro.py` (lines 85-87, 229-254, 179-240)

## Files to Delete (Invalid Results)
- `data/derived/real_fit_numpyro_optimized.csv` ❌ INVALID
- `data/derived/real_fit_numpyro_optimized_logpi.csv` ❌ INVALID
- `data/derived/real_fit_numpyro_optimized_groupplateau.csv` ❌ INVALID
- All previous interpretations and conclusions ❌ INVALID

---

## Verification
Run test: `poetry run python test_corrected_kalman.py`

Expected output:
```
✅ Log-likelihood is in reasonable range!
✅ State converged to expected range!
```
