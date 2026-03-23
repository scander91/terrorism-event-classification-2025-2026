#!/usr/bin/env python3
"""
Step 3: Generate Final Corrected Numbers for Chapter 2
=======================================================
Reads results from steps 1 and 2, produces a definitive report
of CLAIMED vs ACTUAL values with specific LaTeX fix suggestions.

Depends on: step1_raw_profile.json, step2_pipeline_results.json
Output: step3_final_report.txt, step3_latex_fixes.tex
"""

import os
import sys
import json
from datetime import datetime

PROJECT_ROOT = os.path.expanduser("~/TerrorismNER_Project")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ch2_verification_results")


def load_results():
    """Load results from steps 1 and 2."""
    results = {}

    step1_path = os.path.join(OUTPUT_DIR, "step1_raw_profile.json")
    if os.path.exists(step1_path):
        with open(step1_path) as f:
            results["step1"] = json.load(f)
    else:
        print(f"WARNING: {step1_path} not found. Run step 1 first.")

    step2_path = os.path.join(OUTPUT_DIR, "step2_pipeline_results.json")
    if os.path.exists(step2_path):
        with open(step2_path) as f:
            results["step2"] = json.load(f)
    else:
        print(f"WARNING: {step2_path} not found. Run step 2 first.")

    return results


def generate_final_report(results):
    """Generate the master comparison report."""
    lines = []
    lines.append("=" * 78)
    lines.append("  CHAPTER 2 — FINAL VERIFICATION REPORT")
    lines.append(f"  Generated: {datetime.now().isoformat()}")
    lines.append("=" * 78)

    s1 = results.get("step1", {})
    s2 = results.get("step2", {})
    mm = s1.get("missing_metrics", {})
    claimed = s1.get("claimed_values", {})
    snapshots = s2.get("snapshots", [])

    # ═══════════════════════════════════════════════════════════════════
    # SECTION A: CRITICAL NUMERICAL CONFLICTS
    # ═══════════════════════════════════════════════════════════════════
    lines.append(f"\n{'═'*78}")
    lines.append("  A. CRITICAL NUMERICAL CONFLICTS — CLAIMED vs ACTUAL")
    lines.append(f"{'═'*78}")

    lines.append(f"\n  {'#':<4} {'What':<45} {'Claimed':>10} {'Actual':>10} {'Verdict':>10}")
    lines.append(f"  {'-'*4} {'-'*45} {'-'*10} {'-'*10} {'-'*10}")

    issues = []

    # A1: Initial dimensions
    if mm:
        a_samples = mm.get("n_samples", "?")
        a_features = mm.get("n_features", "?")
        ok_s = "✅" if a_samples == 208688 else "❌"
        ok_f = "✅" if a_features == 135 else "❌"
        lines.append(f"  {'A1':<4} {'Initial samples':<45} {'208,688':>10} {str(a_samples):>10} {ok_s:>10}")
        lines.append(f"  {'A2':<4} {'Initial features':<45} {'135':>10} {str(a_features):>10} {ok_f:>10}")

        # A3: Feature-level prevalence
        a_prev = mm.get("eta_features_pct", "?")
        ok = "✅" if isinstance(a_prev, (int, float)) and abs(a_prev - 56.3) < 0.15 else "❌"
        lines.append(f"  {'A3':<4} {'Feature-level prevalence (%)':<45} {'56.3':>10} {str(a_prev):>10} {ok:>10}")
        if ok == "❌":
            issues.append(f"Feature prevalence: claimed 56.3%, actual {a_prev}%")

        # A4: Average missing rate
        a_avg = mm.get("eta_bar_pct", "?")
        ok = "✅" if isinstance(a_avg, (int, float)) and abs(a_avg - 41.2) < 0.5 else "❌"
        lines.append(f"  {'A4':<4} {'Avg missing rate per feature (%)':<45} {'41.2':>10} {str(a_avg):>10} {ok:>10}")
        if ok == "❌":
            issues.append(f"Avg missing rate: claimed 41.2%, actual {a_avg}%")

        # A5: Overall cell-level missing rate
        a_cell = mm.get("eta_overall_pct", "?")
        ok = "✅" if isinstance(a_cell, (int, float)) and abs(a_cell - 38.7) < 0.5 else "❌"
        lines.append(f"  {'A5':<4} {'Cell-level missing rate (%) [Eq 2.4]':<45} {'38.7':>10} {str(a_cell):>10} {ok:>10}")
        if ok == "❌":
            issues.append(f"Cell-level missing: claimed 38.7%, actual {a_cell}%")

        # A6: Table 5 conflation
        lines.append(f"  {'A6':<4} {'Table 5 Overall Missing Rate':<45} {'56.2':>10} {'→WRONG':>10} {'❌ LABEL':>10}")
        issues.append(f"Table 5 labels feature prevalence as 'Overall Missing Rate' and uses 56.2% instead of 56.3%")

    # A7: Data retention
    if snapshots:
        initial_n = snapshots[0].get("samples", 0)
        final_n = snapshots[-1].get("samples", 0)
        actual_retention = round(final_n / initial_n * 100, 1) if initial_n > 0 else "?"
        simple_retention = round(180706 / 208688 * 100, 1)
        lines.append(f"  {'A7':<4} {'Data retention (%)':<45} {'89.7':>10} {str(actual_retention):>10} {'❌' if abs(89.7 - simple_retention) > 0.5 else '✅':>10}")
        issues.append(f"Data retention: 180706/208688 = {simple_retention}%, NOT 89.7%")

    # A8: Post-preprocessing missing rate
    if snapshots:
        # After imputation snapshot
        for snap in snapshots:
            if "imputation" in snap.get("stage", ""):
                post_cell = snap.get("cell_missing_pct", "?")
                lines.append(f"  {'A8':<4} {'Post-imputation cell missing (%)':<45} {'4.8':>10} {str(post_cell):>10} {'?':>10}")

        final_snap = snapshots[-1]
        final_cell = final_snap.get("cell_missing_pct", "?")
        final_feat_prev = final_snap.get("feature_prevalence_pct", "?")
        lines.append(f"  {'A9':<4} {'Final cell missing (%)':<45} {'4.8':>10} {str(final_cell):>10} {'?':>10}")
        lines.append(f"  {'A10':<4} {'Final feature prevalence (%)':<45} {'?':>10} {str(final_feat_prev):>10} {'NEW':>10}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION B: FEATURE COUNT TRACKING
    # ═══════════════════════════════════════════════════════════════════
    lines.append(f"\n{'═'*78}")
    lines.append("  B. FEATURE COUNT TRACKING (Table 4 Verification)")
    lines.append(f"{'═'*78}")

    if snapshots:
        lines.append(f"\n  {'Stage':<35} {'Num':>5} {'Cat':>5} {'Txt':>5} {'Total':>6}")
        lines.append(f"  {'-'*35} {'-'*5} {'-'*5} {'-'*5} {'-'*6}")
        for snap in snapshots:
            lines.append(f"  {snap['stage']:<35} {snap['numerical']:>5} "
                         f"{snap['categorical']:>5} {snap['text']:>5} "
                         f"{snap['total_features']:>6}")

        lines.append(f"\n  Chapter 2 Table 4 claims:")
        lines.append(f"  {'After Redundancy':<35} {'44':>5} {'13':>5} {'15':>5} {'72':>6}")
        lines.append(f"  {'After Feature Engineering':<35} {'40':>5} {'28':>5} {'15':>5} {'83':>6}")
        lines.append(f"  {'Final (After VIF)':<35} {'30':>5} {'28':>5} {'15':>5} {'73':>6}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION C: VIF DETAILS
    # ═══════════════════════════════════════════════════════════════════
    lines.append(f"\n{'═'*78}")
    lines.append("  C. VIF REMOVAL DETAILS (Missing from Chapter)")
    lines.append(f"{'═'*78}")

    vif_details = s2.get("details", {}).get("phase5", {})
    removed_vif = vif_details.get("removed_features", [])
    if removed_vif:
        lines.append(f"\n  Features removed (VIF > {vif_details.get('vif_threshold', 10)}):")
        lines.append(f"  {'#':<4} {'Feature':<35} {'VIF':>10}")
        lines.append(f"  {'-'*4} {'-'*35} {'-'*10}")
        for i, r in enumerate(removed_vif, 1):
            lines.append(f"  {i:<4} {r['feature']:<35} {r['vif']:>10.2f}")
        lines.append(f"\n  Total removed: {len(removed_vif)} (Chapter claims: 10)")
    else:
        lines.append("  No VIF results available. Check step 2 output.")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION D: KL-DIVERGENCE
    # ═══════════════════════════════════════════════════════════════════
    lines.append(f"\n{'═'*78}")
    lines.append("  D. KL-DIVERGENCE VALUES (Missing from Chapter)")
    lines.append(f"{'═'*78}")
    lines.append(f"  Chapter claims D_KL < ε but never specifies ε or reports values.\n")

    kl = s2.get("kl_divergence", {})
    if kl:
        lines.append(f"  {'Feature':<35} {'KL-Divergence':>15}")
        lines.append(f"  {'-'*35} {'-'*15}")
        for col, info in sorted(kl.items(), key=lambda x: x[1]["kl_divergence"], reverse=True):
            lines.append(f"  {col:<35} {info['kl_divergence']:>15.6f}")

        import numpy as np
        kl_vals = [v["kl_divergence"] for v in kl.values()]
        lines.append(f"\n  Mean KL-divergence: {np.mean(kl_vals):.6f}")
        lines.append(f"  Max KL-divergence:  {np.max(kl_vals):.6f}")
        lines.append(f"  Suggested ε:        {np.max(kl_vals) * 1.5:.4f} (1.5× max observed)")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION E: ALL ISSUES SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    lines.append(f"\n{'═'*78}")
    lines.append("  E. COMPLETE ISSUES LIST — ACTION REQUIRED")
    lines.append(f"{'═'*78}")

    for i, issue in enumerate(issues, 1):
        lines.append(f"  {i}. {issue}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION F: CORRECTED VALUES FOR THESIS
    # ═══════════════════════════════════════════════════════════════════
    lines.append(f"\n{'═'*78}")
    lines.append("  F. CORRECTED VALUES TO USE IN THESIS")
    lines.append(f"{'═'*78}")

    if mm:
        lines.append(f"\n  Use these ACTUAL values (replace any conflicting claims):\n")
        lines.append(f"  Initial dataset:          {mm.get('n_samples', '?'):,} samples × "
                     f"{mm.get('n_features', '?')} features")
        lines.append(f"  η_features (Eq 2.2):      {mm.get('eta_features_pct', '?')}% "
                     f"({mm.get('features_with_any_missing', '?')}/{mm.get('n_features', '?')})")
        lines.append(f"  η̄ (Eq 2.3):               {mm.get('eta_bar_pct', '?')}%")
        lines.append(f"  η_overall (Eq 2.4):       {mm.get('eta_overall_pct', '?')}%")
        lines.append(f"  Data retention:            {180706}/{mm.get('n_samples', '?')} = "
                     f"{round(180706/mm['n_samples']*100, 1) if mm.get('n_samples') else '?'}%")

    if snapshots:
        final = snapshots[-1]
        lines.append(f"  Final features:           {final.get('total_features', '?')} "
                     f"(Num:{final.get('numerical', '?')}, "
                     f"Cat:{final.get('categorical', '?')}, "
                     f"Txt:{final.get('text', '?')})")
        lines.append(f"  Final cell missing:       {final.get('cell_missing_pct', '?')}%")
        lines.append(f"  Final feature prevalence: {final.get('feature_prevalence_pct', '?')}%")
        lines.append(f"  Final complete rows:      {final.get('complete_rows_pct', '?')}%")

    lines.append(f"\n{'═'*78}")
    lines.append("  END OF REPORT")
    lines.append(f"{'═'*78}\n")

    report = "\n".join(lines)
    report_path = os.path.join(OUTPUT_DIR, "step3_final_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    return report


def generate_latex_fixes(results):
    """Generate specific LaTeX replacement suggestions."""
    s1 = results.get("step1", {})
    mm = s1.get("missing_metrics", {})

    lines = []
    lines.append("% ================================================================")
    lines.append("% CHAPTER 2 — CORRECTED LaTeX VALUES")
    lines.append(f"% Generated: {datetime.now().isoformat()}")
    lines.append("% Replace the corresponding sections in your thesis")
    lines.append("% ================================================================")

    if mm:
        eta_f = mm.get("eta_features_pct", "XX.X")
        n_missing = mm.get("features_with_any_missing", "XX")
        n_features = mm.get("n_features", "XXX")
        eta_bar = mm.get("eta_bar_pct", "XX.X")
        eta_overall = mm.get("eta_overall_pct", "XX.X")
        n_samples = mm.get("n_samples", "XXX,XXX")

        lines.append("")
        lines.append("% ── FIX 1: Introduction statistics ────────────────────────────────")
        lines.append(f"% With {eta_f}\\% of features ({n_missing} out of {n_features})")
        lines.append(f"% exhibiting missing values, an average missing rate of {eta_bar}\\%")
        lines.append(f"% per feature, and an overall cell-level missing rate of {eta_overall}\\%")
        lines.append("")
        lines.append("% ── FIX 2: Table 5 — Split into two rows ─────────────────────────")
        lines.append("% REPLACE the single 'Overall Missing Rate' row with:")
        lines.append(f"Feature-Level Prevalence $\\eta_{{\\text{{features}}}}$ & "
                     f"{eta_f}\\% & [COMPUTE]\\% & [COMPUTE]\\% \\\\")
        lines.append(f"Overall Missing Rate $\\eta_{{\\text{{overall}}}}$ & "
                     f"{eta_overall}\\% & [COMPUTE]\\% & [COMPUTE]\\% \\\\")
        lines.append("")
        lines.append("% ── FIX 3: Data Retention ────────────────────────────────────────")
        if isinstance(n_samples, int):
            ret = round(180706 / n_samples * 100, 1)
            lines.append(f"% Data Retention = 180,706 / {n_samples:,} = {ret}\\%")
            lines.append(f"% NOT 89.7\\% as currently stated")
        lines.append("")
        lines.append("% ── FIX 4: Section 2.3.2 — Remove repetition ────────────────────")
        lines.append("% REPLACE the verbose repetition with:")
        lines.append("% As established in Section~\\ref{Chapter 2}, the GTD's extensive")
        lines.append("% missing data (see Equations~\\ref{eq:feature_prevalence}--")
        lines.append("% \\ref{eq:overall_missing_rate}) and substantial noise in incident")
        lines.append("% reporting necessitate systematic preprocessing.")
        lines.append("")
        lines.append("% ── FIX 5: Consistent rounding ──────────────────────────────────")
        lines.append(f"% Use {eta_f}\\% everywhere (NOT 56.2\\%)")
        lines.append(f"% 76/135 = 0.56296... ≈ {eta_f}\\%")

    tex_path = os.path.join(OUTPUT_DIR, "step3_latex_fixes.tex")
    content = "\n".join(lines)
    with open(tex_path, "w") as f:
        f.write(content)

    print(f"\nLaTeX fixes saved: {tex_path}")
    return content


def main():
    print(f"{'='*70}")
    print(f"  STEP 3: FINAL CORRECTED NUMBERS")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")

    results = load_results()

    if not results:
        print("ERROR: No results found. Run steps 1 and 2 first.")
        sys.exit(1)

    generate_final_report(results)
    generate_latex_fixes(results)

    print(f"\n{'='*70}")
    print(f"  All reports saved to: {OUTPUT_DIR}")
    print(f"  Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath) / 1024
        print(f"    {f:<40} ({size:.1f} KB)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
