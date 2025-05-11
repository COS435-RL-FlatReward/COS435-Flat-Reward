import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--env_id", type=str, default="Ant-v4")
    parser.add_argument("--ppo_run_name", type=str, default="ppo_continuous_action")
    parser.add_argument("--ppo_seed", type=int, default=1)
    parser.add_argument("--sam_run_name", type=str, default="ppo_continuous_action_sam")
    parser.add_argument("--sam_seed", type=int, default=1)
    parser.add_argument("--sam_rho", type=float, default=0.005)
    args = parser.parse_args()

    results_dir = args.results_dir
    env_id = args.env_id
    ppo_run_name = args.ppo_run_name
    ppo_seed = args.ppo_seed
    
    sam_run_name = args.sam_run_name
    sam_seed = args.sam_seed
    sam_rho = args.sam_rho
    
    # # Load the data
    # action_noise_std_list = np.load(f'results/{run_name}_action_noise_levels.npy')
    # avgs_action_noise = np.load(f'results/{run_name}_action_noise_avgs.npy')
    # stds_action_noise = np.load(f'results/{run_name}_action_noise_stds.npy')

    # Load the data for PPO
    ppo_action_noise_std_list = np.load(os.path.join(results_dir, f"{env_id}__{ppo_run_name}__{ppo_seed}_action_noise_levels.npy"))
    ppo_avgs_action_noise = np.load(os.path.join(results_dir, f"{env_id}__{ppo_run_name}__{ppo_seed}_action_noise_avgs.npy"))
    ppo_stds_action_noise = np.load(os.path.join(results_dir, f"{env_id}__{ppo_run_name}__{ppo_seed}_action_noise_stds.npy"))
    # Load the data for SAM
    sam_action_noise_std_list = np.load(os.path.join(results_dir, f"{env_id}__{sam_run_name}__{sam_seed}__rho{sam_rho}_action_noise_levels.npy"))
    sam_avgs_action_noise = np.load(os.path.join(results_dir, f"{env_id}__{sam_run_name}__{sam_seed}__rho{sam_rho}_action_noise_avgs.npy"))
    sam_stds_action_noise = np.load(os.path.join(results_dir, f"{env_id}__{sam_run_name}__{sam_seed}__rho{sam_rho}_action_noise_stds.npy"))

    # Plot the results
    plt.figure(figsize=(5, 3))
    plt.plot(ppo_action_noise_std_list, ppo_avgs_action_noise, label='PPO', marker='o')
    plt.fill_between(ppo_action_noise_std_list, ppo_avgs_action_noise - ppo_stds_action_noise, ppo_avgs_action_noise + ppo_stds_action_noise, alpha=0.2)
    plt.plot(sam_action_noise_std_list, sam_avgs_action_noise, label=f'SAM+PPO (rho={sam_rho})', marker='o')
    plt.fill_between(sam_action_noise_std_list, sam_avgs_action_noise - sam_stds_action_noise, sam_avgs_action_noise + sam_stds_action_noise, alpha=0.2)
    plt.title(f"Action Noise Robustness Evaluation for {env_id}")
    plt.xlabel("Action Noise Standard Deviation")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{env_id}_{ppo_run_name}_vs_{sam_run_name}_action_noise_plot.png"))
    
    # Load the friction data for PPO
    ppo_friction_std_list = np.load(os.path.join(results_dir, f"{env_id}__{ppo_run_name}__{ppo_seed}_friction_levels.npy"))
    ppo_avgs_friction = np.load(os.path.join(results_dir, f"{env_id}__{ppo_run_name}__{ppo_seed}_friction_avgs.npy"))
    ppo_stds_friction = np.load(os.path.join(results_dir, f"{env_id}__{ppo_run_name}__{ppo_seed}_friction_stds.npy"))
    # Load the friction data for SAM
    sam_friction_std_list = np.load(os.path.join(results_dir, f"{env_id}__{sam_run_name}__{sam_seed}__rho{sam_rho}_friction_levels.npy"))
    sam_avgs_friction = np.load(os.path.join(results_dir, f"{env_id}__{sam_run_name}__{sam_seed}__rho{sam_rho}_friction_avgs.npy"))
    sam_stds_friction = np.load(os.path.join(results_dir, f"{env_id}__{sam_run_name}__{sam_seed}__rho{sam_rho}_friction_stds.npy"))
    # Plot the results
    plt.figure(figsize=(5, 3))
    plt.plot(ppo_friction_std_list, ppo_avgs_friction, label='PPO', marker='o')
    plt.fill_between(ppo_friction_std_list, ppo_avgs_friction - ppo_stds_friction, ppo_avgs_friction + ppo_stds_friction, alpha=0.2)
    plt.plot(sam_friction_std_list, sam_avgs_friction, label=f'SAM+PPO (rho={sam_rho})', marker='o')
    plt.fill_between(sam_friction_std_list, sam_avgs_friction - sam_stds_friction, sam_avgs_friction + sam_stds_friction, alpha=0.2)
    plt.title(f"Friction Robustness Evaluation for {env_id}")
    plt.xlabel("Friction Coefficient Standard Deviation")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{env_id}_{ppo_run_name}_vs_{sam_run_name}_friction_plot.png"))
    
    # Load the mass data for PPO
    ppo_mass_std_list = np.load(os.path.join(results_dir, f"{env_id}__{ppo_run_name}__{ppo_seed}_mass_levels.npy"))
    ppo_avgs_mass = np.load(os.path.join(results_dir, f"{env_id}__{ppo_run_name}__{ppo_seed}_mass_avgs.npy"))
    ppo_stds_mass = np.load(os.path.join(results_dir, f"{env_id}__{ppo_run_name}__{ppo_seed}_mass_stds.npy"))
    # Load the mass data for SAM
    sam_mass_std_list = np.load(os.path.join(results_dir, f"{env_id}__{sam_run_name}__{sam_seed}__rho{sam_rho}_mass_levels.npy"))
    sam_avgs_mass = np.load(os.path.join(results_dir, f"{env_id}__{sam_run_name}__{sam_seed}__rho{sam_rho}_mass_avgs.npy"))
    sam_stds_mass = np.load(os.path.join(results_dir, f"{env_id}__{sam_run_name}__{sam_seed}__rho{sam_rho}_mass_stds.npy"))
    # Plot the results
    plt.figure(figsize=(5, 3))
    plt.plot(ppo_mass_std_list, ppo_avgs_mass, label='PPO', marker='o')
    plt.fill_between(ppo_mass_std_list, ppo_avgs_mass - ppo_stds_mass, ppo_avgs_mass + ppo_stds_mass, alpha=0.2)
    plt.plot(sam_mass_std_list, sam_avgs_mass, label=f'SAM+PPO (rho={sam_rho})', marker='o')
    plt.fill_between(sam_mass_std_list, sam_avgs_mass - sam_stds_mass, sam_avgs_mass + sam_stds_mass, alpha=0.2)
    plt.title(f"Mass Robustness Evaluation for {env_id}")
    plt.xlabel("Mass Standard Deviation")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{env_id}_{ppo_run_name}_vs_{sam_run_name}_mass_plot.png"))
    
    # Load the friction and mass data for PPO
    ppo_avgs_friction_mass = np.load(os.path.join(results_dir, f"{env_id}__{ppo_run_name}__{ppo_seed}_friction_mass_avgs.npy"))
    ppo_stds_friction_mass = np.load(os.path.join(results_dir, f"{env_id}__{ppo_run_name}__{ppo_seed}_friction_mass_stds.npy"))
    # Load the friction and mass data for SAM
    sam_avgs_friction_mass = np.load(os.path.join(results_dir, f"{env_id}__{sam_run_name}__{sam_seed}__rho{sam_rho}_friction_mass_avgs.npy"))
    sam_stds_friction_mass = np.load(os.path.join(results_dir, f"{env_id}__{sam_run_name}__{sam_seed}__rho{sam_rho}_friction_mass_stds.npy"))

    fric_fractions = ppo_friction_std_list
    mass_fractions = ppo_mass_std_list
    ppo_avg_matrix = ppo_avgs_friction_mass.reshape(len(fric_fractions), len(mass_fractions))
    sam_avg_matrix = sam_avgs_friction_mass.reshape(len(fric_fractions), len(mass_fractions))
    # Plot the results as heatmap with a shared colorbar
    # plt.subplot(1, 2, 1)
    # c = plt.pcolormesh(fric_fractions, mass_fractions, ppo_avg_matrix, shading='auto', cmap='viridis')
    # plt.colorbar(c)
    # plt.xlabel('Mass Factor', fontsize=20)
    # plt.ylabel('Friction Factor', fontsize=20)
    # plt.title(f'PPO', fontsize=20)
    # plt.subplot(1, 2, 2)
    # c = plt.pcolormesh(fric_fractions, mass_fractions, sam_avg_matrix, shading='auto', cmap='viridis')
    # plt.colorbar(c)
    # plt.xlabel('Mass Factor', fontsize=20)
    # plt.ylabel('Friction Factor', fontsize=20)
    # plt.title(f'SAM+PPO (rho={sam_rho})', fontsize=20)
    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir, f"{env_id}_{ppo_run_name}_vs_{sam_run_name}_friction_mass_plot.png"))

    plt.figure(figsize=(8, 4))
    names = [f'PPO', f'SAM+PPO (rho={sam_rho})']
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    v_min = np.inf
    v_max = -np.inf
    for data in [ppo_avg_matrix, sam_avg_matrix]:
        v_min = min(v_min, np.min(data))
        v_max = max(v_max, np.max(data))
    for i, data in enumerate([ppo_avg_matrix, sam_avg_matrix]):
        im = axes[i].imshow(data, vmin=v_min, vmax=v_max)
        axes[i].set_title(f"{names[i]}")
        axes[i].set_xlabel('Mass factor')
        axes[i].set_ylabel('Friction factor')
        axes[i].set_xticks(np.arange(0, 11), np.round(np.arange(0.5, 1.51, 0.1), 1))
        axes[i].set_yticks(np.arange(0, 11), np.round(np.arange(0.375, 1.626, 0.125), 1))
        # axes[i].grid()
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    plt.suptitle(f'Mass and Friction Robustness Evaluation (Env: {env_id}, seed: {ppo_seed})')
    plt.savefig(os.path.join(results_dir, f"{env_id}_{ppo_run_name}_vs_{sam_run_name}_friction_mass_plot.png"))