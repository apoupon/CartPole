import gym
import argparse

from src.ppo import ppo
from src import utils


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='CartPole-v1')
parser.add_argument('--hid', type=int, default=64)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--cpu', type=int, default=1)
parser.add_argument('--steps', type=int, default=4000)
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--use_wandb', type=bool, default=True)
parser.add_argument('--render', type=bool, default=False)

args = parser.parse_args()


if args.use_wandb:
    import wandb
    wandb.init(project='CartPole')
    
if args.render:
    ppo(lambda: gym.make(args.env, render_mode = 'human'), actor_critic=utils.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, use_wandb=args.use_wandb)
else:
    ppo(lambda: gym.make(args.env), actor_critic=utils.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, use_wandb=args.use_wandb)