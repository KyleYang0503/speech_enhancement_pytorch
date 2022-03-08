import argparse
from network import darcn
from config import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
from data import TrainDataset, CvDataset, TrainDataLoader, CvDataLoader
from solver import Solver
from loss import numParams

# fix random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)


def main(args, model):
    tr_dataset = TrainDataset(json_dir=args.json_path,
                              batch_size=args.batch_size)
    cv_dataset = CvDataset(json_dir=args.json_path,
                           batch_size=args.cv_batch_size)
    tr_loader = TrainDataLoader(data_set=tr_dataset,
                                batch_size=1,
                                num_workers=args.num_workers)
    cv_loader = CvDataLoader(data_set=cv_dataset,
                             batch_size=1,
                             num_workers=args.num_workers)
    data= {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    # count the parameter number of the network
    print('The number of trainable parameters of the net is:%d' % (numParams(model)))
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.l2)
    solver = Solver(data, model, optimizer, args)
    solver.train()
    

parser = argparse.ArgumentParser(
    "Speech Enhancement networks with dynamic attention and recursive learning"
)

# parameters config
parser.add_argument('--json_path', type=str, default=json_path,
                    help='The directory of the dataset feat,json format')
parser.add_argument('--loss_path', type=str, default=loss_path,
                    help='The directory to save tr loss and cv loss')
parser.add_argument('--batch_size', type=int, default=batch_size,
                    help='The number of the batch size')
parser.add_argument('--cv_batch_size', type=int, default=batch_size,
                    help='The number of the batch size')
parser.add_argument('--epochs', type=int, default=epoch,
                    help='The number of the training epoch')
parser.add_argument('--lr', type=float, default=lr,
                    help='Initialized learning rate of the network')
parser.add_argument('--early_stop', dest='early_stop', default=1, type=int,
                    help='Early stop training when no improvement for 5 epochs')
parser.add_argument('--half_lr', type=int, default=1,
                    help='Whether to decay learning rate to half scale')
parser.add_argument('--shuffle', type=int, default=1,
                    help='Whether to shuffle within each batch')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers to generate batch')
parser.add_argument('--l2', type=float, default=1e-7,
                    help='weight decay (L2 penalty)')
parser.add_argument('--save_path', type=str, default=save_path,
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=check_point, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='./Model/epoch19.pth.tar', #'./Model/idling/q3/epoch.10pth.tar'
                    help='Continue from checkpoint model')
parser.add_argument('--best_path', default=best_path,
                    help='Location to save best cv model')
parser.add_argument('--print_freq', type=int, default=5000,
                    help='The frequency of printing loss infomation')

train_model = darcn(causal_flag=causal_flag, stage_number=stage_number)

if __name__ == '__main__':
    args = parser.parse_args()
    model = train_model
    print(args)
    main(args, model)