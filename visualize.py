import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os 
import gym 
import argparse
import datetime

offline_hyperparameters = {
    'halfcheetah-medium-v2':         {'lr': 3e-4, 'alpha': 1.0, 'eta': 2.0,   'num_epochs': 2000, 'gn': 9.0,  'expectile': 0.7},
    'halfcheetah-medium-replay-v2':  {'lr': 3e-4, 'alpha': 1.0, 'eta': 2.0,   'num_epochs': 2000, 'gn': 2.0,  'expectile': 0.7},
    'halfcheetah-medium-expert-v2':  {'lr': 3e-4, 'alpha': 1.0, 'eta': 1.0,   'num_epochs': 2000, 'gn': 8.0,  'expectile': 0.7},
    'hopper-medium-v2':              {'lr': 3e-4, 'alpha': 1.0, 'eta': 2.0,   'num_epochs': 2000, 'gn': 9.0,  'expectile': 0.6},
    'hopper-medium-replay-v2':       {'lr': 3e-4, 'alpha': 1.0, 'eta': 1.0,   'num_epochs': 2000, 'gn': 4.0,  'expectile': 0.6},
    'hopper-medium-expert-v2':       {'lr': 3e-4, 'alpha': 1.0, 'eta': 0.5,   'num_epochs': 2000, 'gn': 5.0,  'expectile': 0.6},
    'walker2d-medium-v2':            {'lr': 3e-4, 'alpha': 1.0, 'eta': 1.0,   'num_epochs': 2000, 'gn': 1.0,  'expectile': 0.6},
    'walker2d-medium-replay-v2':     {'lr': 3e-4, 'alpha': 1.0, 'eta': 1.0,   'num_epochs': 2000, 'gn': 4.0,  'expectile': 0.6},
    'walker2d-medium-expert-v2':     {'lr': 3e-4, 'alpha': 1.0, 'eta': 1.0,   'num_epochs': 2000, 'gn': 5.0,  'expectile': 0.6},
    'antmaze-umaze-v0':              {'lr': 3e-4, 'alpha': 1.0, 'eta': 1.0,   'num_epochs': 1000, 'gn': 2.0,  'expectile': 0.9},
    'antmaze-umaze-diverse-v0':      {'lr': 3e-4, 'alpha': 1.0, 'eta': 2.0,   'num_epochs': 1000, 'gn': 3.0,  'expectile': 0.9},
    'antmaze-medium-play-v0':        {'lr': 1e-3, 'alpha': 1.0, 'eta': 4.0,   'num_epochs': 1000, 'gn': 2.0,  'expectile': 0.9},
    'antmaze-medium-diverse-v0':     {'lr': 3e-4, 'alpha': 1.0, 'eta': 3.0,   'num_epochs': 1000, 'gn': 1.0,  'expectile': 0.9},
    'antmaze-large-play-v0':         {'lr': 3e-4, 'alpha': 1.0, 'eta': 4.5,   'num_epochs': 1000, 'gn': 10.0, 'expectile': 0.9},
    'antmaze-large-diverse-v0':      {'lr': 3e-4, 'alpha': 1.0, 'eta': 3.5,   'num_epochs': 1000, 'gn': 7.0,  'expectile': 0.9},
    'pen-human-v1':                  {'lr': 3e-5, 'alpha': 1.0, 'eta': 0.15,  'num_epochs': 1000, 'gn': 7.0,  'expectile': 0.7},
    'pen-cloned-v1':                 {'lr': 3e-5, 'alpha': 1.0, 'eta': 0.1,   'num_epochs': 1000, 'gn': 8.0,  'expectile': 0.7},                 
}

online_hyperparameters = {
    'mujoco': {'lr': 3e-4, 'alpha': 0.05, 'eta': 1.0, 'num_epochs': 1000, 'gn': 2.0},
    'dmc':    {'lr': 3e-4, 'alpha': 0.05, 'eta': 1.0, 'num_epochs': 500 , 'gn': 2.0},
}

def visualize_agent(env, state_dim, action_dim, device, args, actor_dir):
    from agents.ql_cm import CPQL as Agent

    # Initialize the agent
    agent = Agent(state_dim=state_dim,
                  action_dim=action_dim,
                  action_space=env.action_space,
                  rl_type=args.rl_type,
                  device=device,
                  discount=args.discount,
                  max_q_backup=args.max_q_backup,
                  lr=args.lr,
                  eta=args.eta,
                  alpha=args.alpha,
                  lr_decay=args.lr_decay,
                  lr_maxt=args.num_epochs,
                  grad_norm=args.gn,
                  q_mode=args.q_mode,
                  sampler=args.sampler,
                  sample_num=args.sample_num,
                  TD_sample=args.TD_sample,
                  expectile=args.expectile,
                  memory_size=args.memory_size,
                  clip_denoised=False
                  )

    # Load the actor model from the checkpoint directory
    checkpoint_path = os.path.join(actor_dir, 'actor_1000.pth')
    agent.actor.load_state_dict(torch.load(checkpoint_path))
    agent.actor.eval()  # Set the actor to evaluation mode

    # Sample a random state from the environment
    # Check if state space has infinite bounds and handle it
    low, high = env.observation_space.low, env.observation_space.high

    # If there are infinite bounds, replace them with some large values
    low = np.where(np.isinf(low), -10.0, low)
    high = np.where(np.isinf(high), 10.0, high)

    # Sample a state from the modified state space
    sampled_state = np.random.uniform(low, high)  # Random state within the bounds
    sampled_state = torch.tensor(sampled_state, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Sample actions using the agent's diffusion model
    actions = agent.diffusion.sample(model=agent.actor, state=sampled_state, sampler=args.sampler)

    # If actions are multi-dimensional, reduce them to 2D for visualization (if necessary)
    if action_dim > 2:
        actions = actions[:, :2]  # Just take the first two dimensions of the action for visualization

    
    # Create a scatter plot of the actions (distribution)
    plt.figure(figsize=(8, 6))
    # Detach the actions tensor from the computation graph and convert to numpy for plotting
    actions = actions.detach().cpu().numpy()
    print(actions)
    print(sampled_state)


    # Use PCA to reduce dimensionality of actions to 2 dimensions
    pca = PCA(n_components=2)
    actions_2d = pca.fit_transform(actions)  # Apply PCA

    # Plot the reduced actions (2D visualization)
    plt.figure(figsize=(8, 6))
    plt.scatter(actions_2d[:, 0], actions_2d[:, 1], alpha=0.5, s=10)
    plt.title(f'Actions sampled from Consistency Policy Q Learning')
    plt.xlabel('action dim 1')
    plt.ylabel('action dim 2')
    plt.grid(True)

    # Ensure the visualization directory exists
    visualization_dir = "visualization"
    os.makedirs(visualization_dir, exist_ok=True)

    # Save the plot with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f"{args.env_name}_PCA_actions_{args.sampler}_samplenum_{args.sample_num}_{timestamp}.png"
    plot_filepath = os.path.join(visualization_dir, plot_filename)
    plt.savefig(plot_filepath)

    print(f"Saved PCA-reduced action distribution plot to {plot_filepath}")
    plt.close()  # Close the plot to free up memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--rl_type', default="online", type=str, help='offline or online RL tasks (default: offline)')
    parser.add_argument("--q_mode", default="q", type=str, help='q for CPQL and q_v for CPIQL')
    parser.add_argument("--env_name", default="Hopper-v3", type=str, help='Mujoco Gym environment')
    parser.add_argument("--seed", default=0, type=int, help='random seed (default: 0)')
    parser.add_argument("--dir", default="/home/sherman/Desktop/Consistency/results/online/Hopper-v3/QL|1847|alpha-0.05|eta-1.0|sampler_onestep_quasi_monte_carlo|test_qnorm/", type=str)
    parser.add_argument('--save_checkpoints', action='store_true')
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--online_start_steps", default=10000, type=int)
    parser.add_argument("--memory_size", default=1e6, type=int)
    parser.add_argument("--batch_size", default=256, type=int, help='batch size (default: 256)')
    parser.add_argument("--lr_decay", action='store_true')
    parser.add_argument("--discount", default=0.99, type=float, help='discount factor for reward (default: 0.99)')
    parser.add_argument("--sampler", default="onestep_quasi_monte_carlo", help="the type of sampler used, include onestep montecarlo, onestep multi sample montecarlo and one step multisample quasi monte carlo")
    parser.add_argument("--sample_num", default=2048, type=int, help="the number of samples used in the experiments")
    parser.add_argument("--TD_sample", default=False, type=bool, help="whether to use sampling in computing the target value for TD learning")
    # wandb log
    parser.add_argument("--group", default="Quasi-CPQL-dev", type=str)
    args = parser.parse_args()

    if args.rl_type == 'online' and args.q_mode == 'q_v':
        raise AssertionError("CPIQL is not supported for online RL tasks!")

    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'

    if args.rl_type == 'offline':
        args.num_epochs = offline_hyperparameters[args.env_name]['num_epochs']
        args.lr = offline_hyperparameters[args.env_name]['lr']
        args.eta = offline_hyperparameters[args.env_name]['eta']
        args.alpha = offline_hyperparameters[args.env_name]['alpha']
        args.gn = offline_hyperparameters[args.env_name]['gn']
        args.expectile = offline_hyperparameters[args.env_name]['expectile']

        if 'antmaze' in args.env_name:
            args.max_q_backup = True
            args.reward_tune = 'cql_antmaze'
            #args.sampler = 'multistep'
        else:
            args.max_q_backup = False
            args.reward_tune = 'no'
            #args.sampler = 'onestep'

        args.eval_freq = 50
        args.eval_episodes = 10 if 'v2' in args.env_name else 100
    else: 
        args.num_epochs = online_hyperparameters['mujoco']['num_epochs']
        args.lr = online_hyperparameters['mujoco']['lr']
        args.eta = online_hyperparameters['mujoco']['eta']
        args.alpha = online_hyperparameters['mujoco']['alpha']
        args.gn = online_hyperparameters['mujoco']['gn']
        args.expectile = 0

        args.max_q_backup = False
        args.reward_tune = 'no'
        #args.sampler = 'onestep'
    
        args.eval_freq = 10
        args.eval_episodes = 3 

    # Setup environment
    env = gym.make(args.env_name)

    # Run the visualization
    visualize_agent(env, state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], device=args.device, args=args, actor_dir=args.dir)
