import argparse

ENTITY = 'da25s003-indian-institute-of-technology-madras'
PROJECT = 'DA6401_Assignment1v5'

BEST_MODEL_NPY = 'best_model.npy'
BEST_MODEL_CONFIG = 'best_config.json'

# Hyper-parameter constants used across the codebase
CONFIG = {
   'val_split': 0.2,
   'beta'     : 0.9,
   'epsilon'  : 1e-8,
}

def parse_arguments():
    p = argparse.ArgumentParser(description='DA6401 Assignment-1 — MLP')

    p.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    p.add_argument('-nhl', '--num_layers', type=int, default=3, help='Number of hidden layers')
    p.add_argument('-sz', '--hidden_size',  type=int,  nargs='+', default=[128, 128, 64], help='Neurons per hidden layer.')
    p.add_argument('-a', '--activation',   type=str,  default='relu',
                   choices=['relu', 'tanh', 'sigmoid'])
    p.add_argument('-w_i', '--weight_init',  type=str,  default='xavier',
                   choices=['random', 'xavier', 'zeros'])
    p.add_argument('-e', '--epochs',  type=int,  default=50)
    p.add_argument('-b', '--batch_size',   type=int,  default=32)
    p.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    p.add_argument('-o', '--optimizer',type=str,  default='momentum', choices=['sgd', 'momentum', 'nag', 'rmsprop'])
    p.add_argument('-l', '--loss',    type=str,  default='cross_entropy', choices=['mse', 'cross_entropy'])
    p.add_argument('-wd', '--weight_decay', type=float, default=0.0)

    # hyperparameters 
    p.add_argument('--beta',  type=float, default=0.9, help='Beta for momentum/RMSProp')
    p.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon for numerical stability')

    #  W&B 
    p.add_argument('-w_p', '--wandb_project', type=str,  default=PROJECT)
    p.add_argument('--wandb_entity', type=str,  default=ENTITY)
    p.add_argument('--no_wandb', action='store_true')

    #  Paths 
    p.add_argument('--save_dir', type=str, default='models')
    p.add_argument('--model_save_path', type=str, default=BEST_MODEL_NPY)
    p.add_argument('--model_path', type=str, default=None)
    p.add_argument('--config_path',type=str, default=BEST_MODEL_CONFIG)

    #  Misc 
    p.add_argument('--experiment', type=str, default='train', choices=['train', 'visual', 'sweep', 'optimizer', 'vanishing', 'dead', 'loss', 'overlay', 'error', 'symmetry', 'fashion', 'all'])
    p.add_argument('--high_lr', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)

    #  Data split for validation 
    p.add_argument('--val_split', type=float, default=0.1, help='Validation set fraction')
    p.add_argument('--val_fraction', type=float, default=None, help='Validation set fraction (alias for val_split)')

    args = p.parse_args()
    # Normalize hidden_size handling
    if len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size * args.num_layers
    elif len(args.hidden_size) != args.num_layers:
        hs = args.hidden_size
        args.hidden_size = (hs + [hs[-1]] * args.num_layers)[:args.num_layers]

    if args.val_fraction is None:
        args.val_fraction = args.val_split

    return args