import argparse

def add_general_group(group):
    group.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    group.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    group.add_argument("--save-path", type=str, default="results/", help="dir path for output file")
    group.add_argument("--seed", type=int, default=42, help="seed value")
    group.add_argument("--mode", type=str, default='train', help="Mode of running")
    group.add_argument("--train_mode", type=str, default='clean', help="Mode of training [clean, dp]")


def add_data_group(group):
    group.add_argument('--dataset', type=str, default='MIR', help="used dataset")
    group.add_argument('--data_path', type=str, default='Data/MIR', help="the directory used to save dataset")



def add_model_group(group):
    group.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    group.add_argument("--lr", type=float, default=0.01, help="learning rate")
    group.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    group.add_argument('--batch_size', type=int, default=2708)
    group.add_argument('--dropout', type=float, default=0.5)
    group.add_argument('--train_verbose', action='store_true', help="print training details")
    group.add_argument('--log_every', type=int, default=1, help='print every x epoch')
    group.add_argument('--eval_every', type=int, default=5, help='evaluate every x epoch')
    group.add_argument('--model_save_path', type=str, default='../SavedModel/')
    group.add_argument("--num_steps", type=int, default=5000)
    group.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")



def add_atk_group(group):
    group.add_argument("--train_attacker", type=bool, default=False)
    group.add_argument("--save_attack_path", type=str, default='../SavedModel/attack/')
    group.add_argument("--attack_training_round", type=int, default=100)
    group.add_argument("--attack_model_path", type=str, default='../SavedModel/attack/')


def add_dp_group(group):
    group.add_argument('--eps_ldp', type=float, default=1.0, help="privacy budget for LDP")
    group.add_argument('--num_bit', type=int, default=10, help="number of bit to use")
    group.add_argument('--exponent_bit', type=int, default=10, help="number of bit to use for the integer part")


def parse_args():
    parser = argparse.ArgumentParser()
    data_group = parser.add_argument_group(title="Data-related configuration")
    model_group = parser.add_argument_group(title="Model-related configuration")
    atk_group = parser.add_argument_group(title="Attack-related configuration")
    general_group = parser.add_argument_group(title="General configuration")
    defense_group = parser.add_argument_group(title="Defense configuration")

    add_data_group(data_group)
    add_model_group(model_group)
    add_atk_group(atk_group)
    add_general_group(general_group)
    add_dp_group(defense_group)
    return parser.parse_args()
