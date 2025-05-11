# Instructions for implementation

## SAM training  
`python train_sam_ppo.py --config configs/Hopper.yaml --use_sam`

## PPO training  
`python train_ppo.py --config configs/Hopper.yaml`

## SAM Eval
`python eval_action_sam_ppo.py --config configs/Hopper.yaml --use_sam`

`python eval_fricmass_sam_ppo.py --config configs/Hopper.yaml --use_sam`

## PPO Eval
`python eval_action_sam_ppo.py --config configs/Hopper.yaml`

`python eval_fricmass_sam_ppo.py --config configs/Hopper.yaml`

## Drawing plots

go to `./analysis.ipynb` and check out the codes.