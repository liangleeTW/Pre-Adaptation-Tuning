# Figure Mapping for Overleaf Upload

## Figures to Upload to Overleaf `figures/` Folder

### Main Text Figures (Currently Referenced in paper.txt)

1. **Figure 1**: `main_comparison_m1exp_vs_mtwoR.png`
   - **Label**: `fig:main_comparison`
   - **Referenced in**: Results §3.2
   - **Description**: 8-panel comprehensive comparison showing WAIC, R², RMSE, bias, β posteriors, example fits, and residuals

2. **Figure 2**: `state_slopes_vs_deltalogpi.png`
   - **Label**: `fig:state_slopes`
   - **Referenced in**: Results §3.3
   - **Description**: State trajectory validation (r = -0.23, p = 0.056) vs error-based slopes (r = 0.02, p = 0.90)

3. **Figure 3**: `ppc_m_obs_adaptation_curves.png`
   - **Label**: `fig:ppc`
   - **Referenced in**: Results §3.1
   - **Description**: Posterior predictive checks showing M2-dual fits for all three groups

4. **Figure 4**: `group_comparisons.png`
   - **Label**: `fig:group_comparison`
   - **Referenced in**: Results §3.4
   - **Description**: Group-level parameter distributions (β_state, β_obs, R_state, R_obs)

---

## Tables in paper.txt (No upload needed - already in LaTeX)

1. **Table 1**: Model Comparison Statistics
   - **Label**: `tab:model_comparison`
   - **Referenced in**: Results §3.1
   - **Shows**: WAIC, ΔW AIC, R², RMSE, late bias, R̂ for all 6 models

2. **Table 2**: Group-Level Parameter Estimates
   - **Label**: `tab:group_parameters`
   - **Referenced in**: Results §3.2
   - **Shows**: β_state, β_obs, R_state, R_obs by group with HDIs

---

## Available But Not Currently Referenced

### Could Be Added to Main Text:

- **`r_obs_vs_late_variability.png`**
  - Shows R_obs doesn't correlate with empirical late-trial variability (r = 0.10, p = 0.40)
  - **Supports claim in §3.2 paragraph on β_obs**
  - **Recommendation**: Add as Figure 5 or move to Supplement

- **`beta_posteriors_by_group.png`**
  - Standalone version of panels E & F from main_comparison
  - **Recommendation**: Supplement only (redundant with Fig 1)

### Potentially Supplementary:

- `model_comparison_waic.png` - Bar chart of WAIC (redundant with Table 1)
- `residuals_by_trial.png` - Residual analysis (partially in Fig 1 panel H)
- `slope_distribution_by_group.png` - Learning slope distributions
- `ppc_adaptation_curves_by_group.png` - May be M1-exp fits?
- `empirical_slope_vs_deltalogpi.png` - Redundant with Fig 2 right panel

---

## Missing Visualizations That Should Be Created

### High Priority (Strengthen Main Claims):

1. ✅ **ALREADY EXISTS**: R_obs vs late variability
   - File: `r_obs_vs_late_variability.png`
   - Just needs to be referenced in paper

2. **Example subject fits (detailed)**
   - Show 3-4 individual subjects with varying Δπ values
   - Demonstrate how M2-dual captures individual variation
   - **Panel**: Observed errors + M2-dual state trajectory + predictions
   - **Currently**: Partially shown in main_comparison panel G (only 1 subject)

3. **Kalman gain evolution**
   - Show how Kalman gain K_t evolves across trials
   - Compare subjects with high vs low Δπ
   - **Demonstrates**: Mechanistic link between β_state and learning speed

### Medium Priority (Supplementary Material):

4. **Convergence diagnostics**
   - Trace plots for key parameters (β_state, β_obs)
   - R-hat evolution across chains
   - **Purpose**: Show MCMC sampling quality

5. **Prior vs posterior comparison**
   - Show prior distributions overlaid with posteriors
   - **Purpose**: Demonstrate data informativeness

6. **Scatter: Predicted vs observed state slopes**
   - X-axis: Predicted slope from β_state × Δπ
   - Y-axis: Empirical state slope
   - **Purpose**: Validate β_state quantitatively

---

## Recommendation

### For Main Text:
- **Keep current 4 figures** (comprehensive and clear)
- **Add Figure 5**: r_obs_vs_late_variability.png (supports β_obs interpretation)
- **Create Figure 6**: Individual subject examples (3-4 subjects × 3 panels each)

### For Supplement:
- Create Kalman gain figure
- Create convergence diagnostics
- Move beta_posteriors_by_group.png
- Move residuals_by_trial.png

---

## Next Steps

1. ✅ Current figures are already created and ready
2. ⏳ Add r_obs figure reference to paper.txt
3. ⏳ Create individual subject fits figure
4. ⏳ Create Kalman gain evolution figure
5. Upload all to Overleaf figures/ folder
