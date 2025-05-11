import os
from typing import Callable
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

def evaluate_action_robustness(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    gamma: float = 0.99,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    max_action = envs.single_action_space.high[0]
    print(f"Max action: {max_action}")
    
    action_noise_std_list = np.linspace(0.0, 0.5, 11)  # From 0.0 to 0.5 with 11 steps
    avgs_action_noise = []
    stds_action_noise = []
    episodic_returns = []
    
    for action_noise_std in action_noise_std_list:
        rewards = []
        for _ in range(eval_episodes):
            obs, _ = envs.reset(seed=np.random.randint(1000))
            end = False
            while not end:
                actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
                noise = torch.normal(0, action_noise_std, size=actions.shape).to(device)
                noisy_action = actions + noise
                # noisy_action = torch.clamp(noisy_action, -max_action, max_action)
                next_obs, _, _, _, infos = envs.step(noisy_action.cpu().numpy())
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if "episode" in info:
                            avg_reward = info["episode"]["r"]
                            end = True
                obs = next_obs
            rewards.append(avg_reward)
            
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print("---------------------------------------")
        print(f'Action Noise STD: {action_noise_std}: Avg Reward: {avg_reward:.3f}, Std: {std_reward:.3f}')
        print("---------------------------------------")
        avgs_action_noise.append(avg_reward)
        stds_action_noise.append(std_reward)
        
        if action_noise_std == 0.0:
            episodic_returns = rewards
    
    os.makedirs('results', exist_ok=True)
    np.save(f'results/{run_name}_action_noise_levels', action_noise_std_list)
    np.save(f'results/{run_name}_action_noise_avgs', avgs_action_noise)
    np.save(f'results/{run_name}_action_noise_stds', stds_action_noise)
    
    
    # while len(episodic_returns) < eval_episodes:
    #     actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
    #     next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
    #     if "final_info" in infos:
    #         for info in infos["final_info"]:
    #             if "episode" not in info:
    #                 continue
    #             print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
    #             episodic_returns += [info["episode"]["r"]]
    #     obs = next_obs

    
    
    return episodic_returns


def evaluate_friction_mass_robustness(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    gamma: float = 0.99,
    fric_fractions = np.linspace(0.4, 1.6, 11),
    mass_fractions = np.linspace(0.5, 1.5, 11),
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    max_action = envs.single_action_space.high[0]
    print(f"Max action: {max_action}")
    
    # Retrieve original mass and friction values
    # print("\n=== Retrieving Original Mass and Friction Values ===")
    # Print all the atribute names of the model
    # print("=========================================")
    # print(envs.envs[0].unwrapped.model.geom('left_leg_geom'))
    # print(envs.envs[0].unwrapped.model.geom('left_leg_geom').id)
    # print(envs.envs[0].unwrapped.model.geom('left_leg_geom').friction)
    # print(envs.envs[0].unwrapped.model.body('front_left_leg'))
    # print(envs.envs[0].unwrapped.model.body('front_left_leg').mass)
    # print(envs.envs[0].unwrapped.model.body_mass)
    # print(envs.envs[0].unwrapped.model.geom_bodyid)
    # print(envs.envs[0].unwrapped.model.geom_friction)
    # print("=========================================")
    
    # ======================================================
    # Define the friction and mass parameters for different environments
    # ======================================================
    
    if env_id in ['HalfCheetah-v4', 'Walker2d-v4', 'Hopper-v4', 'Ant-v4', 'Swimmer-v4', 'Humanoid-v4']:
        if env_id == 'HalfCheetah-v4':
            # Define friction and mass parameters for HalfCheetah
            # fric_fractions = np.linspace(0.2, 0.8, 11)  
            # fric_bodies = ['fthigh'] 
            fric_bodies = ['ffoot']   
            mass_bodies = ['torso'] 

        elif env_id == 'Walker2d-v4':
            # Define friction and mass parameters for Walker2d  
            fric_bodies = ['foot_left'] 
            # fric_bodies = ['foot', 'foot_left']  
            mass_bodies = ['torso']  

        elif env_id == 'Hopper-v4':
            # Define friction and mass parameters for Hopper 
            fric_bodies = ['foot']    
            mass_bodies = ['torso']  
        
        elif env_id == 'Ant-v4':
            # Define friction and mass parameters for Ant  
            fric_bodies = ['floor'] #, 'right_leg_geom'
            mass_bodies = ['torso']
            # fric_body_names = ['aux_1_geom', 
            #                    'aux_2_geom', 
            #                    'aux_3_geom', 
            #                    'aux_4_geom', 
            #                    'back_leg_geom', 
            #                    'floor', 
            #                    'fourth_ankle_geom', 
            #                    'left_ankle_geom', 
            #                    'left_leg_geom',
            #                    'right_ankle_geom',
            #                    'right_leg_geom',
            #                    'rightback_leg_geom',
            #                    'third_ankle_geom', 
            #                    'torso_geom']
            # mass_body_names = ['aux_1',
            #                    'aux_2',
            #                    'aux_3',
            #                    'aux_4',
            #                    'back_leg',
            #                    'front_left_leg',
            #                    'front_right_leg',
            #                    'right_back_leg',
            #                    'torso',
            #                    'world']
            
        elif env_id == 'Swimmer-v4':
            # Define friction and mass parameters for Swimmer  
            fric_bodies = ['back']    
            mass_bodies = ['torso']
            
        elif env_id == 'Humanoid-v4':
            # Define friction and mass parameters for Humanoid   
            fric_bodies = ['floor']    
            mass_bodies = ['left_thigh']
            # fric_body_names = ['butt', 
            #                    'floor', 
            #                    'head', 
            #                    'left_foot', 
            #                    'left_hand', 
            #                    'left_larm', 
            #                    'left_shin1', 
            #                    'left_thigh1', 
            #                    'left_uarm1', 
            #                    'lwaist', 
            #                    'right_foot',
            #                    'right_hand', 
            #                    'right_larm', 
            #                    'right_shin1', 
            #                    'right_thigh1', 
            #                    'right_uarm1', 
            #                    'torso1', 
            #                    'uwaist']
            # mass_body_names = ['left_foot', 
            #                    'left_lower_arm', 
            #                    'left_shin', 
            #                    'left_thigh', 
            #                    'left_upper_arm', 
            #                    'lwaist', 
            #                    'pelvis', 
            #                    'right_foot', 
            #                    'right_lower_arm', 
            #                    'right_shin', 
            #                    'right_thigh', 
            #                    'right_upper_arm', 
            #                    'torso', 
            #                    'world']

        original_masses = {}
        for body in mass_bodies:
            body_id = envs.envs[0].unwrapped.model.body(body).id
            original_masses[body_id] = envs.envs[0].unwrapped.model.body(body_id).mass
        
        original_frictions = {}
        for body in fric_bodies:
            body_id = envs.envs[0].unwrapped.model.geom(body).bodyid
            geom_ids = [i for i, geom_body in enumerate(envs.envs[0].unwrapped.model.geom_bodyid) if geom_body == body_id]
            original_frictions[body] = {geom_id: envs.envs[0].unwrapped.model.geom_friction[geom_id].copy() for geom_id in geom_ids}
        
    else:
        raise NotImplementedError(f"Environment {env_id} not supported for friction and mass robustness evaluation.")
    
    # ======================================================
    # Evaluate the friction robustness
    # ======================================================
    avgs_friction = []
    stds_friction = []
    print("\n=== Evaluating Friction Robustness ===")
    for f in fric_fractions:
        for body in fric_bodies:
            body_id = envs.envs[0].unwrapped.model.geom(body).bodyid
            geom_ids = [i for i, geom_body in enumerate(envs.envs[0].unwrapped.model.geom_bodyid) if geom_body == body_id]
            rewards = []
            
            for _ in range(eval_episodes):
                # Set the friction for the body
                for geom_id in geom_ids:
                    envs.envs[0].unwrapped.model.geom_friction[geom_id] = [f, f, 0.1] * original_frictions[body][geom_id]
                    # envs.envs[0].unwrapped.model.geom_friction[geom_id] = f * original_frictions[body][geom_id]
                    
                # Reset the environment
                obs, _ = envs.reset(seed=np.random.randint(1000))
                
                end = False
                while not end:
                    actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
                    next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
                    if "final_info" in infos:
                        for info in infos["final_info"]:
                            if "episode" in info:
                                avg_reward = info["episode"]["r"]
                                end = True
                    obs = next_obs
                rewards.append(avg_reward)
                
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            print("---------------------------------------")
            print(f'Friction Fraction: {f}: Avg Reward: {avg_reward:.3f}, Std: {std_reward:.3f}')
            print("---------------------------------------")
            avgs_friction.append(avg_reward)
            stds_friction.append(std_reward)
    
    # Save the friction results
    os.makedirs('results', exist_ok=True)
    np.save(f'results/{run_name}_friction_levels', fric_fractions)
    np.save(f'results/{run_name}_friction_avgs', avgs_friction)
    np.save(f'results/{run_name}_friction_stds', stds_friction)
    
    # ======================================================
    # Evaluate the mass robustness
    # ======================================================
    avgs_mass = []
    stds_mass = []
    print("\n=== Evaluating Mass Robustness ===")
    for m in mass_fractions:
        for body in mass_bodies:
            body_id = envs.envs[0].unwrapped.model.body(body).id
            rewards = []
            
            for _ in range(eval_episodes):
                # Set the mass for the body
                envs.envs[0].unwrapped.model.body_mass[body_id] = m * original_masses[body_id]
                
                # Reset the environment
                obs, _ = envs.reset(seed=np.random.randint(1000))
                
                end = False
                while not end:
                    actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
                    next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
                    if "final_info" in infos:
                        for info in infos["final_info"]:
                            if "episode" in info:
                                avg_reward = info["episode"]["r"]
                                end = True
                    obs = next_obs
                rewards.append(avg_reward)
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            print("---------------------------------------")
            print(f'Mass Fraction: {m}: Avg Reward: {avg_reward:.3f}, Std: {std_reward:.3f}')
            print("---------------------------------------")
            avgs_mass.append(avg_reward)
            stds_mass.append(std_reward)
    # Save the mass results
    os.makedirs('results', exist_ok=True)
    np.save(f'results/{run_name}_mass_levels', mass_fractions)
    np.save(f'results/{run_name}_mass_avgs', avgs_mass)
    np.save(f'results/{run_name}_mass_stds', stds_mass)
    
    # =======================================================
    # Evaluate the friction and mass robustness together
    # =======================================================
    avgs_friction_mass = []
    stds_friction_mass = []
    print("\n=== Evaluating Friction and Mass Robustness ===")
    for f in fric_fractions:
        for m in mass_fractions:
            for body_fric, body_mass in zip(fric_bodies, mass_bodies):
                body_fric_id = envs.envs[0].unwrapped.model.geom(body_fric).bodyid
                geom_ids = [i for i, geom_body in enumerate(envs.envs[0].unwrapped.model.geom_bodyid) if geom_body == body_fric_id]
                rewards = []
                
                for _ in range(eval_episodes):
                    # Set the friction and mass for the body
                    for geom_id in geom_ids:
                        envs.envs[0].unwrapped.model.geom_friction[geom_id] = [f, f, 0.1] * original_frictions[body_fric][geom_id]
                        # envs.envs[0].unwrapped.model.geom_friction[geom_id] = f * original_frictions[body_fric][geom_id]
                    body_mass_id = envs.envs[0].unwrapped.model.body(body_mass).id
                    envs.envs[0].unwrapped.model.body_mass[body_mass_id] = m * original_masses[body_mass_id]
                    
                    # Reset the environment
                    obs, _ = envs.reset(seed=np.random.randint(1000))
                    
                    end = False
                    while not end:
                        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
                        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
                        if "final_info" in infos:
                            for info in infos["final_info"]:
                                if "episode" in info:
                                    avg_reward = info["episode"]["r"]
                                    end = True
                        obs = next_obs
                    rewards.append(avg_reward)
                avg_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                print("---------------------------------------")
                print(f'Friction Fraction: {f}, Mass Fraction: {m}: Avg Reward: {avg_reward:.3f}, Std: {std_reward:.3f}')
                print("---------------------------------------")
                avgs_friction_mass.append(avg_reward)
                stds_friction_mass.append(std_reward)
    # Save the friction and mass results
    os.makedirs('results', exist_ok=True)
    np.save(f'results/{run_name}_friction_mass_avgs', avgs_friction_mass)
    np.save(f'results/{run_name}_friction_mass_stds', stds_friction_mass)
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_names", 
                       nargs='+',  # This allows multiple arguments
                       type=str,   # Each argument is a string
                       required=True, # Makes the argument required
                       help="List of run names to evaluate")
    args = parser.parse_args()
    for run_name in args.run_names:
        exp_name = run_name.split("__")[1]
        env_id = run_name.split("__")[0]
        sam_append = ""
        if "sam" in exp_name:
            sam_rho = run_name.split("__")[2]
            sam_append = f"__{sam_rho}"
        seed = int(run_name.split("__")[-1])
        # time = run_name.split("__")[-1]
        
        model_path = f"runs/{run_name}/{exp_name}.cleanrl_model"
        evaluate_action_robustness(
            model_path,
            make_env,
            env_id,
            eval_episodes=10,
            run_name=f"{env_id}__{exp_name}__{seed}{sam_append}",
            Model=Agent,
            device="cpu",
            capture_video=False,
        )
        
        evaluate_friction_mass_robustness(
            model_path,
            make_env,
            env_id,
            eval_episodes=10,
            run_name=f"{env_id}__{exp_name}__{seed}{sam_append}",
            Model=Agent,
            device="cpu",
            capture_video=False,
            gamma=0.99,
            fric_fractions=np.linspace(0.1, 1.9, 11),
            mass_fractions=np.linspace(0.1, 1.9, 11),
        )
