# ============================================================
# RUN BASELINE — REPLICATION SCRIPT
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from asm_memf_main import train_memf_v7

def run():

    print("\n========================================")
    print("RUNNING BASELINE SIMULATION")
    print("========================================\n")

    # --------------------------------------------------------
    # Create folders
    # --------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # --------------------------------------------------------
    # Run model
    # --------------------------------------------------------
    df_episode, df_agent, df_period, firms, households, banks, government, central_bank, ext_env, G = train_memf_v7()

    print("Simulation finished.\n")

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    df_episode.to_csv("results/episode_results.csv", index=False)
    df_agent.to_csv("results/agent_results.csv", index=False)
    df_period.to_csv("results/period_results.csv", index=False)

    print("Results saved in /results\n")

    # --------------------------------------------------------
    # BASIC FIGURE (for quick validation)
    # --------------------------------------------------------
    plt.figure(figsize=(8,5))

    plt.plot(df_episode["episode"], df_episode["avg_productivity_final"], label="Productivity")
    plt.plot(df_episode["episode"], df_episode["default_rate"], label="Default rate")

    plt.legend()
    plt.title("Baseline Dynamics")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/baseline_dynamics.png", dpi=300)
    plt.close()

    print("Figure saved in /figures\n")

    print("========================================")
    print("BASELINE RUN COMPLETED")
    print("========================================\n")


if __name__ == "__main__":
    run()
