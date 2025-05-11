# COS435-Flat-Reward

Our project aims to reproduce the results of the ICLR25 Oral paper "Flat reward in Policy Parameter Space Implies Robust Reinforcement Learning" 

> **Flat reward in Policy Parameter Space Implies Robust Reinforcement Learning**  
> Hyun Kyu Lee, Sung Whan Yoon  
> **Accepted by: ICLR 2025**
>
> [[ICLR 2025](https://openreview.net/forum?id=4OaO3GjP7k)]

Our main contribution is the implementation of the paper's methods and the evaluation of their performance in various environments with reproducible settings from **CleanRL**. We tested the methods on newest **MuJoCo** environments, including **Ant-v4**, and **Humanoid-v4**. We also provided a detailed setup guide for the environment and dependencies.

### Clean RL setup 

```
conda create -n cleanrl python=3.10
conda activate cleanrl
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

cd cleanrl

pip install -r requirements/requirements-mujoco.txt -U
```

### Clean RL Experiments

Run everything in the cleanrl directory: `bash cleanrl/launch_all.sh`

Run PPO continuous: `python cleanrl/ppo_continuous_action.py`

Run PPO continuous sam: `python cleanrl/ppo_continuous_action_sam.py`

Run robustness evaluation: `python cleanrl/ppo_robustness_eval.py`

Run visualization: `python cleanrl/plot_results.py`


### Original Environment setup 

(using OpenAI gym)
Install dependencies
```
Python 3.9
pip 23.2.1
gym 0.21.0
setuptools 57.5.0
wheel 0.37.0
mujoco-py==2.1.2.14
torch==1.12.1
Install MuJoCo
```
Download the MuJoCo version 2.1 binaries for Linux or OSX.
Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210.

Troubleshooting
```
pip install "cython<3"
pip install "numpy<2"
sudo apt-get install patchelf
pip install pyyaml
```

## Citation

```bibtex
@inproceedings{leeflat,
  title={Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning},
  author={Lee, Hyun Kyu and Yoon, Sung Whan},
  booktitle={The Thirteenth International Conference on Learning Representations}
}