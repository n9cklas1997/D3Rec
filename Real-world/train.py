import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import time
from datetime import timedelta
import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset_load import load_data
from utils_train import *
from utils import *
from models import *

from DiffRecGutter.DiffRec.models.DNN import DNN
from DiffRecGutter.DiffRec.models.gaussian_diffusion import ModelMeanType, GaussianDiffusion
from diffrec_adapter import DiffRecAdapter, DNNNoTime


def main(args, dataset_dir_path, best_model_path):
    print(args)
    print(f'Use {args.device}')
    print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    dataset, train_dataset, valid_dataset, test_dataset, _ = load_data(args, dataset_dir_path)
    sp_train, sp_valid, sp_test = dataset.sp_train, dataset.sp_valid, dataset.sp_test

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    diffusion = GaussianDiffusion(
        mean_type=ModelMeanType.EPSILON,
        noise_schedule=args.noise_schedule,
        noise_scale=args.noise_scale,
        noise_min=args.beta_start,
        noise_max=args.beta_end,
        steps=args.steps,
        device=args.device
    )

    # --- Patch diffusion to include sample_new_interaction ---
    if not hasattr(diffusion, "sample_new_interaction"):
        def sample_new_interaction(self, model, x_0, prob=None, guide_w=0.,
                                   sampling_steps=None, sampling_noise=False):
            steps = sampling_steps if sampling_steps is not None else self.steps
            return self.p_sample(model, x_0, steps=steps, sampling_noise=sampling_noise)
        diffusion.sample_new_interaction = sample_new_interaction.__get__(diffusion)
    # --------------------------------------------------------

    raw_dnn = DNN(
        in_dims=[dataset.num_items, 600, 200],
        out_dims=[200, 600, dataset.num_items],
        emb_size=args.dim_step,
        dropout=args.dropout
    ).to(args.device)

    dnn = DNNNoTime(raw_dnn).to(args.device)

    model = DiffRecAdapter(
        dnn_model=dnn,
        num_steps=args.steps,
        t_emb_dim=128
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_recall, best_epoch = -np.inf, np.inf
    best_test_result, best_ckl = None, None
    start_epoch = 1

    print("Start training")
    print("Using model:", type(model))
    print("Model details:", model)

    start_total = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch - best_epoch >= args.patient:  # early stopping
            print('-' * 18)
            print('Exiting from training early')
            break

        start = time.time()
        loss = train_one_epoch(args, model, diffusion, optimizer, train_loader)
        print(f'Epoch {epoch:>3} - train loss: {loss: >10.4f}. time: {str(timedelta(seconds=int(time.time() - start)))}')

        if epoch % args.eval_freq == 0:
            val_recall = evaluate(args, model, diffusion, valid_loader, sp_valid, sp_train, args.topK,
                                  dataset.item_category, dataset.num_cate)
            print(f'Evaluation validation recall@20: {val_recall:.4f}, time: {str(timedelta(seconds=int(time.time() - start)))}')

            if val_recall > best_recall:
                best_recall = val_recall
                test_results = evaluate(args, model, diffusion, test_loader, sp_test, sp_train + sp_valid,
                                             args.topK, dataset.item_category, dataset.num_cate, is_best=True)

                best_epoch, best_test_result = epoch, test_results
                if args.save_model is True:
                    torch.save(model, best_model_path)
                print(f'Evaluation Best test')
                print_metric_results(args.topK, test_results)

    print('#' * 106, f'Training time: {str(timedelta(seconds=int(time.time() - start_total)))}')
    print(f"Test Metirc At Best Valid Metric, epoch: {best_epoch}")
    print_metric_results(args.topK, best_test_result)


def get_args_parser():
    parser = argparse.ArgumentParser(description="D3Rec", add_help=True)

    ##### Training Setting #####
    parser.add_argument('--seed', default=1, type=int, help="Random seed")
    parser.add_argument('--cuda', default=0, type=int, help="GPU index")
    parser.add_argument('--batch_size', default=400, type=int, help="Batch size")
    parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
    parser.add_argument('--wd', default=0.1, type=float, help="Weight decay")
    parser.add_argument('--epochs', default=2000, type=int, help="Num epochs")
    parser.add_argument('--topK', default=[10, 20], type=int, nargs='+', help="Top k list")
    parser.add_argument('--patient', default=500, type=int, help="Early stop patient.")
    parser.add_argument('--save_model', action='store_true', help="Save model parameters.")
    parser.add_argument('--eval_freq', type=int, default=2)

    ##### Data Setting #####
    parser.add_argument('--dataset_name', default='ml-1m', type=str, help="Dataset name")
    parser.add_argument('--split_ratio', default=[6, 2, 2], type=int, nargs="+", help="Train, Valid, Test split ratio")
    parser.add_argument('--drop_num', default=20, type=int, help="Drop user whose history are less than drop_num")
    parser.add_argument('--test_w_valid', action='store_true', help="True: test with train and valide data. False: test with train data.")

    ##### Model hyper parameter #####
    parser.add_argument('--dims', type=int, default=[600, 200], nargs="+", help="the dims for the classifier: n_item -> latent -> 1")
    parser.add_argument('--dropout', default=0.5, type=float, help="Drop interaction")
    parser.add_argument('--dim_step', default=10, type=int, help="Dimension of denoising step")
    parser.add_argument('--lamda', default=1, type=float, help="Peanlty of Orthogonal and Matching loss")
    parser.add_argument('--w_max', default=1, type=float, help="Range of reweight [w_min ~ w_max]")
    parser.add_argument('--w_min', default=0.2, type=float, help="Range of reweight [w_min ~ w_max]")

    ### Diffusion hyper parameter
    parser.add_argument('--beta_start', default=0.0001, type=float, help="Beta at time 0")
    parser.add_argument('--beta_end', default=0.02, type=float, help="Beta at time T")
    parser.add_argument('--noise_scale', default=0.1, type=float, help="Strength of noise which is added into origin data.")
    parser.add_argument('--steps', default=100, type=int, help="Denosing maximum step. (i.e., T value)")
    parser.add_argument('--snr', action='store_true', help="Use snr reweight or not")
    parser.add_argument('--sampling_steps', default=0, type=int, help="Denosing step at generate new data")
    parser.add_argument('--sampling_noise', action='store_true', help="Apply uncertainty at generate new data. If False, mean is new data. Else, mean + variance is new data.")
    parser.add_argument('--noise_schedule', default="linear-var", type=str, choices=['linear', 'linear-var', 'cosine', 'exp', 'binomial', 'sqrt'], help="Beat noise schedule")

    ### Classifier-free
    parser.add_argument('--drop_div', default=0.1, type=float)
    parser.add_argument('--guide_w', default=0.5, type=float)

    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    set_random_seed(random_seed=args.seed)
    args.device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    dataset_dir_path, best_model_path = get_paths(args)
    main(args, dataset_dir_path, best_model_path)
