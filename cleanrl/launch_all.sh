python ppo_continuous_action.py --env_id Ant-v4 --seed 2 --total_timesteps 2000000
python ppo_continuous_action_sam.py --env_id Ant-v4 --seed 2 --total_timesteps 2000000 --sam_rho 0.01
python ppo_robustness_eval.py --run_names Ant-v4__ppo_continuous_action__2 Ant-v4__ppo_continuous_action_sam__rho0.01__2 
python plot_results.py --env_id Ant-v4 --ppo_seed 2 --sam_seed 2 --sam_rho 0.01

python ppo_continuous_action.py --env_id Humanoid-v4 --total_timesteps 2500000 --clip_vloss True
python ppo_continuous_action_sam.py --env_id Humanoid-v4 --sam_rho 0.008 --total_timesteps 2500000 --clip_vloss True
python ppo_robustness_eval.py --run_names Humanoid-v4__ppo_continuous_action__1 Humanoid-v4__ppo_continuous_action_sam__rho0.008__1
python plot_results.py --env_id Humanoid-v4 --ppo_seed 1 --sam_seed 1 --sam_rho 0.008