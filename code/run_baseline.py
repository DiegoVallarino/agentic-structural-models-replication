# ============================================================
# RUN BASELINE — FULL REPLICATION SCRIPT
# ============================================================

import os
import matplotlib.pyplot as plt

from asm_memf_main import train_memf_v7, run_full_analysis


def run():

    print("\n========================================")
    print("RUNNING BASELINE SIMULATION")
    print("========================================\n")

    # --------------------------------------------------------
    # Ensure folders exist
    # --------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # --------------------------------------------------------
    # Run model
    # --------------------------------------------------------
    print("Step 1: Calling model...\n")

    df_episode, df_agent, df_period, firms, households, banks, government, central_bank, ext_env, G = train_memf_v7()

    print("Step 2: Model finished ✅\n")

    # --------------------------------------------------------
    # Save raw outputs
    # --------------------------------------------------------
    print("Step 3: Saving CSVs...\n")

    df_episode.to_csv("results/episode_results.csv", index=False)
    df_agent.to_csv("results/agent_results.csv", index=False)
    df_period.to_csv("results/period_results.csv", index=False)

    print("CSVs saved ✅\n")

    # --------------------------------------------------------
    # Basic figure (quick validation)
    # --------------------------------------------------------
    print("Step 4: Generating baseline figure...\n")

    plt.figure(figsize=(8,5))
    plt.plot(df_episode["episode"], df_episode["avg_productivity_final"], label="Productivity")
    plt.plot(df_episode["episode"], df_episode["default_rate"], label="Default rate")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.title("Baseline Dynamics")

    plt.tight_layout()
    plt.savefig("figures/baseline_dynamics.png", dpi=300)
    plt.close()

    print("Baseline figure saved ✅\n")

    # --------------------------------------------------------
    # Full analysis (tables + additional figures)
    # --------------------------------------------------------
    print("Step 5: Running full analysis...\n")

    run_full_analysis(df_episode, df_agent, df_period, firms, G)

    print("Full analysis completed ✅\n")

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------
    print("========================================")
    print("BASELINE RUN COMPLETED 🚀")
    print("========================================\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run()
