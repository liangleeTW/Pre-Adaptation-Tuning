"""
Run Bayesian group comparisons for both β_state and β_obs.

This wrapper script runs the existing Bayesian comparison scripts on the final models.
"""

from pathlib import Path
import subprocess
import sys

def run_comparison(script_name: str, model_name: str, posterior_path: Path,
                  output_subdir: str):
    """Run a Bayesian comparison script."""
    output_dir = Path("data/derived/bayesian_analysis") / output_subdir

    print(f"\n{'='*70}")
    print(f"Running {script_name} for {model_name}")
    print(f"{'='*70}\n")

    cmd = [
        sys.executable,
        f"scripts/analysis/{script_name}",
        "--posterior-path", str(posterior_path),
        "--output-dir", str(output_dir),
        "--rope", "0.1"
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"❌ Error running {script_name}")
        return False

    print(f"✅ {script_name} completed successfully")
    return True


def main():
    print("\n" + "="*70)
    print("BAYESIAN GROUP COMPARISONS FOR FINAL MODELS")
    print("="*70)

    # Model configurations
    models = {
        "M2-Dissociation": {
            "posterior": Path("data/derived/posteriors/m2_dissociation_posterior.nc"),
            "output_prefix": "m2_dissociation"
        },
        "M3-DD": {
            "posterior": Path("data/derived/posteriors/m3_dd_posterior.nc"),
            "output_prefix": "m3_dd"
        }
    }

    # Check which posteriors exist
    available_models = {
        name: config for name, config in models.items()
        if config["posterior"].exists()
    }

    if not available_models:
        print("\n❌ No posterior files found!")
        print("Expected locations:")
        for name, config in models.items():
            print(f"  - {config['posterior']}")
        return

    print(f"\nFound posteriors for: {', '.join(available_models.keys())}")

    # Ask user which model to analyze
    print("\nWhich model would you like to analyze?")
    print("1. M2-Dissociation (simpler, best WAIC)")
    print("2. M3-DD (decomposed, mechanistic interpretation)")
    print("3. Both")

    choice = input("\nEnter choice (1/2/3) [default: 2]: ").strip() or "2"

    if choice == "1":
        selected = ["M2-Dissociation"]
    elif choice == "2":
        selected = ["M3-DD"]
    else:
        selected = list(available_models.keys())

    # Run comparisons
    for model_name in selected:
        if model_name not in available_models:
            print(f"\n⚠️  Skipping {model_name} (posterior not found)")
            continue

        config = available_models[model_name]
        output_prefix = config["output_prefix"]

        # Run β_state comparison
        run_comparison(
            "bayesian_beta_state_comparison.py",
            model_name,
            config["posterior"],
            f"{output_prefix}_beta_state"
        )

        # Run β_obs comparison
        run_comparison(
            "bayesian_beta_obs_comparison.py",
            model_name,
            config["posterior"],
            f"{output_prefix}_beta_obs"
        )

    print("\n" + "="*70)
    print("✅ ALL BAYESIAN COMPARISONS COMPLETE")
    print("="*70)

    print("\nResults saved to:")
    print("  data/derived/bayesian_analysis/")
    print("\nGenerated files:")
    print("  - *_summary.csv: Posterior summaries by group")
    print("  - *_pairwise.csv: Pairwise comparison probabilities")
    print("  - *_effect_sizes.csv: Cohen's d effect sizes")
    print("  - *_orderings.csv: Probability of group orderings")
    print("  - *_posteriors.png: Visualization (4 panels)")
    print("  - *_rope.png: ROPE analysis (3 panels)")
    print("  - bayesian_comparison_report.txt: Full text report")


if __name__ == "__main__":
    main()
