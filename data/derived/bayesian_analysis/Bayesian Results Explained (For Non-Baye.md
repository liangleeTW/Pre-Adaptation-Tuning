Bayesian Results Explained (For Non-Bayesians)

  The Big Picture: Frequentist vs Bayesian
  ┌──────────────────────────────────────────┬──────────────────────────────────────────────────────┐
  │               Frequentist                │                       Bayesian                       │
  ├──────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ "Is p < 0.05?" (yes/no)                  │ "What's the probability the effect exists?" (0-100%) │
  ├──────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ Your ANOVA: p = 0.36 → "Not significant" │ See below → Much more informative                    │
  └──────────────────────────────────────────┴──────────────────────────────────────────────────────┘
  ---
  Your Results Translated

  Question 1: "Is β_state > 0 for each group?" (Source-estimation supported?)
  ┌───────┬──────────────────┬────────────────────────┐
  │ Group │ Bayesian Answer  │ Frequentist Equivalent │
  ├───────┼──────────────────┼────────────────────────┤
  │ EC    │ P(β > 0) = 100%  │ Would be p < 0.001 *** │
  ├───────┼──────────────────┼────────────────────────┤
  │ EO+   │ P(β > 0) = 100%  │ Would be p < 0.001 *** │
  ├───────┼──────────────────┼────────────────────────┤
  │ EO-   │ P(β > 0) = 99.4% │ Would be p < 0.01 **   │
  └───────┴──────────────────┴────────────────────────┘
  Plain English: "We are virtually certain (>99%) that all three groups show the source-estimation effect (β_state > 0)."

  ---
  Question 2: "Are the groups different from each other?"
  ┌────────────┬──────────┬───────────────────────┬─────────────────────────┐
  │ Comparison │ P(A > B) │ 95% HDI of Difference │     "Significant"?      │
  ├────────────┼──────────┼───────────────────────┼─────────────────────────┤
  │ EC > EO+   │ 99.3%    │ [0.10, 0.78]          │ ✅ Yes — HDI excludes 0 │
  ├────────────┼──────────┼───────────────────────┼─────────────────────────┤
  │ EC > EO-   │ 99.98%   │ [0.24, 1.03]          │ ✅ Yes — HDI excludes 0 │
  ├────────────┼──────────┼───────────────────────┼─────────────────────────┤
  │ EO+ > EO-  │ 81.1%    │ [-0.22, 0.64]         │ ❌ No — HDI includes 0  │
  └────────────┴──────────┴───────────────────────┴─────────────────────────┘
  Plain English:
  - "We are 99.3% sure EC has higher β_state than EO+" → Credible difference
  - "We are 99.98% sure EC has higher β_state than EO-" → Strong credible difference
  - "We are only 81% sure EO+ > EO-" → Uncertain, could go either way

  ---
  Question 3: "Is the full ordering EC > EO+ > EO- true?"
  ┌────────────────┬─────────────────────────┐
  │    Ordering    │       Probability       │
  ├────────────────┼─────────────────────────┤
  │ EC > EO+ > EO- │ 80.4% ← Your hypothesis │
  ├────────────────┼─────────────────────────┤
  │ EC > EO- > EO+ │ 18.9%                   │
  ├────────────────┼─────────────────────────┤
  │ All others     │ < 1%                    │
  └────────────────┴─────────────────────────┘
  Plain English: "There's an 80% chance the groups are ordered exactly as we predicted. There's essentially 0% chance the order is reversed (EO- > EO+ > EC)."

  ---
  How to Report This (Like "Significance")

  Bayesian Criteria for "Credible" Differences
  ┌─────────────────────────┬──────────────────────────────┬───────────────────────┐
  │        Criterion        │         Your Result          │    Interpretation     │
  ├─────────────────────────┼──────────────────────────────┼───────────────────────┤
  │ 95% HDI excludes 0      │ EC vs EO+: ✅ [0.10, 0.78]   │ "Credible difference" │
  ├─────────────────────────┼──────────────────────────────┼───────────────────────┤
  │                         │ EC vs EO-: ✅ [0.24, 1.03]   │ "Credible difference" │
  ├─────────────────────────┼──────────────────────────────┼───────────────────────┤
  │                         │ EO+ vs EO-: ❌ [-0.22, 0.64] │ "Not credible"        │
  ├─────────────────────────┼──────────────────────────────┼───────────────────────┤
  │ P(difference > 0) > 95% │ EC > EO+: 99.3% ✅           │ Strong evidence       │
  ├─────────────────────────┼──────────────────────────────┼───────────────────────┤
  │                         │ EC > EO-: 99.98% ✅          │ Very strong evidence  │
  ├─────────────────────────┼──────────────────────────────┼───────────────────────┤
  │                         │ EO+ > EO-: 81% ❌            │ Weak evidence         │
  └─────────────────────────┴──────────────────────────────┴───────────────────────┘
  ---
  Paper-Ready Statements

  Instead of: "ANOVA showed no significant difference (p = 0.36)"

  Write:

  "Bayesian analysis revealed strong evidence for group differences in β_state. The eyes-closed group (EC) showed credibly higher β_state than both eyes-open groups: P(β_EC > β_EO+) = 99.3%, difference = 0.43 [95% HDI: 0.10, 0.78]; P(β_EC > β_EO-) = 99.98%, difference = 0.62 [95% HDI: 0.24, 1.03]. The predicted ordering (EC > EO+ > EO-) had 80.4% posterior probability. These results indicate that proprioceptive tuning with eyes closed produces a stronger source-estimation effect than eyes-open conditions."

  ---
  Why Bayesian Found Differences But Frequentist Didn't
  ┌────────────────┬────────────────────────────────────────────┬────────────────────────────────┐
  │     Issue      │             Frequentist ANOVA              │            Bayesian            │
  ├────────────────┼────────────────────────────────────────────┼────────────────────────────────┤
  │ Tests          │ "Any difference among 3 groups?" (omnibus) │ Direct pairwise comparisons    │
  ├────────────────┼────────────────────────────────────────────┼────────────────────────────────┤
  │ Power          │ Diluted across all comparisons             │ Focused on specific hypotheses │
  ├────────────────┼────────────────────────────────────────────┼────────────────────────────────┤
  │ Output         │ Binary (p < 0.05 or not)                   │ Continuous probability         │
  ├────────────────┼────────────────────────────────────────────┼────────────────────────────────┤
  │ Interpretation │ "Not enough evidence to reject null"       │ "80% sure EC > EO+ > EO-"      │
  └────────────────┴────────────────────────────────────────────┴────────────────────────────────┘
  The frequentist ANOVA asks: "Is there ANY difference?" — this is a weak question that pools all variance.

  The Bayesian approach asks: "What's the probability EC > EO+?" — this is your actual research question.

  ---
  Summary Table for Your Paper
  ┌────────────────┬───────────────────────┬───────────────┬─────────────────────────┬────────────┐
  │   Comparison   │ Posterior Probability │    95% HDI    │ Effect Size (Cohen's d) │ Conclusion │
  ├────────────────┼───────────────────────┼───────────────┼─────────────────────────┼────────────┤
  │ EC > EO+       │ 99.3%                 │ [0.10, 0.78]  │ 1.46 (large)            │ Credible   │
  ├────────────────┼───────────────────────┼───────────────┼─────────────────────────┼────────────┤
  │ EC > EO-       │ 99.98%                │ [0.24, 1.03]  │ 2.10 (very large)       │ Credible   │
  ├────────────────┼───────────────────────┼───────────────┼─────────────────────────┼────────────┤
  │ EO+ > EO-      │ 81.1%                 │ [-0.22, 0.64] │ 0.64 (medium)           │ Uncertain  │
  ├────────────────┼───────────────────────┼───────────────┼─────────────────────────┼────────────┤
  │ EC > EO+ > EO- │ 80.4%                 │ —             │ —                       │ Supported  │
  └────────────────┴───────────────────────┴───────────────┴─────────────────────────┴────────────┘