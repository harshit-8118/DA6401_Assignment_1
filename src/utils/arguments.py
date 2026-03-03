import argparse

ENTITY  = 'da25s003-indian-institute-of-technology-madras'
PROJECT = 'DA6401_Assignment1v5'

BEST_MODEL_NPY    = 'best_model.npy'
BEST_MODEL_CONFIG = 'best_config.json'

CONFIG = {
    'val_split' : 0.2,
    'beta': 0.9,
    'epsilon': 1e-8
}

def parse_arguments():
    p = argparse.ArgumentParser(description='DA6401 Assignment-1 — Neural Network')
    p.add_argument('-d', '--dataset',  type=str, default='mnist', choices=['mnist', 'fashion_mnist', 'fashion'])
    p.add_argument('-e',  '--epochs', type=int,   default=50)
    p.add_argument('-b',  '--batch_size', type=int,   default=32)
    p.add_argument('-lr', '--learning_rate',  type=float, default=0.001)
    p.add_argument('-o',  '--optimizer', type=str, default='momentum',  choices=['sgd', 'momentum', 'nag', 'rmsprop'])
    p.add_argument('-l',  '--loss', type=str,   default='cross_entropy', choices=['mse', 'cross_entropy'])
    p.add_argument('-wd', '--weight_decay',   type=float, default=0.0)
    p.add_argument('-sz', '--num_neurons', type=int,   nargs='+', default=[128, 128, 64])
    p.add_argument('-a',  '--activation',  type=str,   default='relu',  choices=['relu', 'tanh', 'sigmoid'])
    p.add_argument('-wi', '--weight_init', type=str,   default='xavier', choices=['random', 'xavier', 'zeros'])
    p.add_argument('--experiment', type=str, default='train',  choices=['train', 'visual', 'sweep', 'optimizer', 'vanishing', 'dead', 'loss', 'overlay', 'error', 'symmetry', 'fashion', 'all'])
    p.add_argument('--high_lr', type=float, default=0.1)
    p.add_argument('--save_dir', type=str, default='models')
    p.add_argument('--model_save_path', type=str, default=BEST_MODEL_NPY)
    p.add_argument('--model_path', type=str, default=None)
    p.add_argument('--config_path', type=str, default=BEST_MODEL_CONFIG)
    p.add_argument('--wandb_project', type=str, default=PROJECT)
    p.add_argument('--wandb_entity', type=str,   default=ENTITY)
    p.add_argument('--beta',  type=float, default=0.9)
    p.add_argument('--epsilon', type=float, default=1e-8)
    p.add_argument('--val_fraction', type=float, default=0.2)
    p.add_argument('--no_wandb', action='store_true')
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()