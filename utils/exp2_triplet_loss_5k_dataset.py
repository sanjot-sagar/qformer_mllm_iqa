# Code for: Finetuning code for InternLM MLLM model. But finetuning with contrastive loss now.
#
# Adapted from: exp2_1_train
# Created Date: 16 April 2024
# Last Modified Date: 16 April 2024
# Last Modified Author: Shika

# Common Imports
import json
import time
import datetime
import traceback
import argparse
import math
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import gc

# From our files
from util_dataload import *
from util_networks import get_internLM_model, NormalRegressor1, ComplexMultiheadAttentionAggregator, BasicMultiheadAttentionAggregator
from exp2_test import exp2_2_test_function
from util_get_internlm_logits import get_init_logits, get_sentence_logits
from utils.custom_blended_ordering_loss import FeatureBlendedOrderingLoss

import logging
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


# Training the model class
class DNNIter(nn.Module):
    def __init__(self, config):
        super(DNNIter, self).__init__()

        self.config = config
        self.lr_scheduler = config.lr_scheduler
        self.cawr_restart_iter = config.cawr_restart_iter
        self.lwca_warmup_iter = config.lwca_warmup_iter

        # Set device
        torch.cuda.set_device(self.config.default_device)
        self.device = f"cuda:{self.config.default_device}" if torch.cuda.is_available(
        ) else "cpu"

        # Self-Attention Model init
        if self.config.network_type == 'b_attn':
            self.aggregator = BasicMultiheadAttentionAggregator(
                embed_dim=self.config.embed_dim, num_heads=self.config.num_heads, regressor_bool=False).to(self.device)
        elif self.config.network_type == 'c_attn':
            self.aggregator = ComplexMultiheadAttentionAggregator(
                embed_dim=self.config.embed_dim, num_heads=self.config.num_heads, regressor_bool=False).to(self.device)
        self.aggregator = self.weight_mode(self.aggregator, trainable=True)
        self.aggregator.train()
        self.aggregator.to(self.device)
        self.convert_models_to_fp32(self.aggregator)

        # Regressor Head init
        self.regressor = NormalRegressor1(
            embed_dim=self.config.embed_dim, pool=False).to(self.device)
        self.regressor = self.weight_mode(self.regressor, trainable=True)
        self.regressor.train()
        self.regressor.to(self.device)
        self.convert_models_to_fp32(self.regressor)

        # InternLM Model init, already put to device in function itself
        self.internlm_trainable_params = self.config.internlm_trainable_params
        self.model = get_internLM_model(self.config)
        self.model = self.internlm_weight_mode(
            self.model, self.internlm_trainable_params, trainable=True)
        self.model.train()
        self.convert_models_to_fp32(self.model)

        # Setting randomness
        torch.manual_seed(self.config.random_seed)
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        # Identify the trainable parameters from self.model and self.aggregrator and self.regressor for optimization
        optimizable_parameters = []
        self.internlm_optimizable_params = []
        models = [self.aggregator, self.model, self.regressor]
        for model in models:
            for name, param in model.named_parameters():
                if model == self.aggregator or model == self.regressor:
                    optimizable_parameters.append((name, param))
                    print(name)
                elif any([param_name in name for param_name in self.config.optimizer_params]):
                    if param.requires_grad:  # Use 'param.requires_grad' for checking
                        optimizable_parameters.append((name, param))
                        self.internlm_optimizable_params.append((name, param))
                        print(name)
                else:
                    param.requires_grad_(False)

        if self.config.optim == "adamw":
            self.optim = AdamW([param for _, param in optimizable_parameters], lr=self.config.lr,
                               weight_decay=self.config.weight_decay)  # note: only give trainable parameters to the optimizer
        elif self.config.optim == "sgd":
            self.optim = torch.optim.SGD([param for _, param in optimizable_parameters], lr=self.config.lr,
                                         weight_decay=self.config.weight_decay)

        self.test_dict = {}
        self.test_srocc = {'iteration': [], 'srocc': []}

        # Setting up results folder
        run_number = len(os.listdir(config.results_dir))
        self.curr_result_dir = os.path.join(
            config.results_dir, f'Run{run_number:04}')
        if not os.path.exists(self.curr_result_dir):
            os.mkdir(self.curr_result_dir)
        self.config.results_dir = self.curr_result_dir

        # Dumping config details to folder
        config_details_path = os.path.join(
            self.config.results_dir, 'config_details.json')
        json_object = json.dumps(self.config.__dict__, indent=4)
        with open(config_details_path, "w") as outfile:
            outfile.write(json_object)

        self.logger = SummaryWriter(
            (Path(self.curr_result_dir) / 'Logs').as_posix())
        self.save_flag = True

        # Setting up the huggingface parameters for sentence generation
        # self.gen_config = dict(
        #     num_beams=1,
        #     do_sample=False,
        #     min_length=1,
        #     repetition_penalty=1.5,
        #     length_penalty=1.0,
        #     temperature=1.0,
        #     max_new_tokens=15,
        #     output_hidden_states=True,
        #     return_dict_in_generate=True,
        # )
        # self.config.gen_config = self.gen_config

        return

    # Below 2 functions for setting model to trainable or not
    @staticmethod
    def weight_mode(model, trainable=True):
        for name, param in model.named_parameters():
            if trainable:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return model

    @staticmethod
    def internlm_weight_mode(model, trainable_params, trainable=True):
        for name, param in model.named_parameters():
            if any([param_name in name for param_name in trainable_params]):
                if trainable:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                param.requires_grad_(False)
        return model

    # just in case converting model weights to float32
    def convert_models_to_fp32(self, model):
        for p in model.parameters():
            p.data = p.data.float()
        return

    # Below 2 functions for getting the data for training and testing
    def init_dataloaders(self):
        train_data, test_data = get_livefb_annotation_data(self.config)
        processed_train_data = CustomTrainDatasetSyntheticLIVEFB(
            df_data=train_data, synthetic_img_dir=self.config.synthetic_img_dir)
        processed_test_data = CustomDataset(
            df_data=test_data, img_dir=self.config.img_dir)

        self.pooled_train_loader = DataLoader(
            dataset=processed_train_data, batch_size=self.config.batch_size, shuffle=True)
        self.pooled_test_loader = DataLoader(
            dataset=processed_test_data, batch_size=self.config.test_batch_size, shuffle=False)

        return

    @staticmethod
    def get_next_batch(dataloader, iterator):
        try:
            next_batch = next(iterator)
        except StopIteration:
            print("Stop iteration encountered.")
            iterator = iter(dataloader)
            next_batch = next(iterator)
        return next_batch, iterator

    # Below 2 functions for getting network training hyperparameters
    def get_scheduler(self, total_iterations):
        total_iterations = total_iterations
        if self.lr_scheduler == 'cosine_annealing':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optim, T_max=total_iterations, eta_min=1e-7)
        if self.lr_scheduler == 'cosine_annealing_warm_restart':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=self.optim, T_0=self.cawr_restart_iter)

        # if self.lr_scheduler == 'linear_warmup_cosine_annealing':
        #     lr_lambda = (
        #         lambda cur_iter: cur_iter / self.lwca_warmup_iter
        #         if cur_iter <= self.lwca_warmup_iter
        #         else 0.5 * (1 + math.cos(math.pi * (cur_iter - self.lwca_warmup_iter) / (self.num_iterations - self.lwca_warmup_iter)))
        #     )
        #     return torch.optim.lr_scheduler.LambdaLR(
        #         self.optim, lr_lambda=lr_lambda)
        return None

    @staticmethod
    def update_learning_rate(optimizer, factor):
        for group in optimizer.param_groups:
            group['lr'] *= factor
        return

    # Below 2 functions for saving trained model and loading it too for later
    def save_model(self, model, aggregator, regressor, optimizer, best=False):
        model_ckpt_path = Path(self.config.results_dir) / 'Train'
        if not os.path.exists(model_ckpt_path):
            os.mkdir(model_ckpt_path)

        if best:
            model_ckpt_path = os.path.join(model_ckpt_path, 'best.tar')
        else:
            model_ckpt_path = os.path.join(
                model_ckpt_path, f'iter_{self.current_iteration}.tar')

        # For internlm model, save only the trainable parameters as it's a huge model
        parameters = {name: param.data for name,
                      param in self.internlm_optimizable_params}

        save_model = {'state_dict': parameters}
        save_aggregator = {'state_dict': aggregator.state_dict()}
        save_regressor = {'state_dict': regressor.state_dict()}
        save_opt = {'state_dict': optimizer.state_dict()}
        full_dict = {'model': save_model, 'optimizer': save_opt, 'regressor': save_regressor,
                     'current_iteration': self.current_iteration, 'aggregator': save_aggregator}

        torch.save(full_dict, model_ckpt_path)
        return

    def load_model(self, path):
        print("Loading model to continue training")
        checkpoint = torch.load(path)
        self.model.load_state_dict(
            checkpoint['model']['state_dict'], strict=False)
        self.aggregator.load_state_dict(checkpoint['aggregator']['state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer']['state_dict'])
        self.regressor.load_state_dict(checkpoint['regressor']['state_dict'])
        print("regressor loaded")
        self.current_iteration = checkpoint['current_iteration']
        return

    # Below 2 functions for actually training the model
    def get_loss_fn(self, predictions, ground_truth):
        predictions = predictions.squeeze(-1)
        try:
            ground_truth = ground_truth.squeeze(-1)
        except:
            pass
        if self.config.loss == 'mse':
            return torch.nn.functional.mse_loss(predictions, ground_truth)
        elif self.config.loss == 'l1':
            return torch.nn.functional.l1_loss(predictions, ground_truth)

    def train_model(self):
        train_loss = []
        self.current_iteration = 1
        self.init_dataloaders()
        iterator_model = iter(self.pooled_train_loader)
        criterion_contrastive = FeatureBlendedOrderingLoss()

        self.test_dict['test_srocc'] = {'srocc_value': [], 'iter_no': []}

        start_iteration = 1
        total_iterations = int(
            (self.config.epochs * len(self.pooled_train_loader)))
        test_iteration = int(
            (self.config.test_epoch * len(self.pooled_train_loader)))

        if self.config.resume_training == True:
            self.load_model(self.config.resume_model_path)
            start_iteration = self.current_iteration + 1
            self.test_dict['test_srocc']['iter_no'].append(
                self.current_iteration)
            self.test_during_train()

        self.model = self.internlm_weight_mode(
            self.model, self.internlm_trainable_params, trainable=True)
        self.model.train()
        self.aggregator = self.weight_mode(self.aggregator, trainable=True)
        self.aggregator.train()
        self.regressor = self.weight_mode(self.regressor, trainable=True)
        self.regressor.train()

        if self.config.scheduler == True:
            scheduler = self.get_scheduler(total_iterations)

        for iteration in tqdm(range(start_iteration, total_iterations + 1)):

            # if iteration == 1:
            #     self.test_dict['test_srocc']['iter_no'].append(
            #         self.current_iteration)
            #     self.test_during_train()
            #     self.model = self.internlm_weight_mode(
            #         self.model, self.internlm_trainable_params, trainable=True)
            #     self.model.train()
            #     self.aggregator = self.weight_mode(
            #         self.aggregator, trainable=True)
            #     self.aggregator.train()
            #     self.regressor = self.weight_mode(
            #         self.regressor, trainable=True)
            #     self.regressor.train()

            sampled_batch, iterator_model = self.get_next_batch(
                self.pooled_train_loader, iterator_model)
            dist_img_input = sampled_batch['img'].to(
                self.device)  # bs, 5, c, h, w
            mos_target = sampled_batch['mos'].to(self.device)

            (b, d, c, h, w) = dist_img_input.shape
            dist_grouped = (dist_img_input.reshape(b * d, c, h, w)).to("cuda")

            # how to get hidden states from internlm model
            if self.config.logit_processing_type == 'init':
                hidden_states_dists = get_init_logits(self.model, dist_grouped)
            elif self.config.logit_processing_type == 'sentence':
                hidden_states_dists = get_sentence_logits(
                    self.model, dist_grouped, self.gen_config)

            # Our self-attention model
            attention_output_dists = self.aggregator(hidden_states_dists)

            # Do feature level contrastive learning on the attention outputs of distorted versions of images
            attention_output_dists = attention_output_dists.reshape(b, d, -1)
            loss_contrastive = criterion_contrastive(attention_output_dists)

            # Get MSE loss on regressed attention outputs of reference images
            # ref image is first image from every batch of dists images
            attention_output_refs = attention_output_dists[:, 0, :]
            scores = self.regressor(attention_output_refs)
            if scores.shape != mos_target.shape:
                scores.squeeze_()
                mos_target.squeeze_()
            loss_mse = self.get_loss_fn(scores, mos_target)

            loss = loss_mse + self.config.alpha_scaling * loss_contrastive

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if self.config.scheduler == True:
                scheduler.step()

            # Logging to tensorboard, log optimizer also
            # is gradient flow being stopped by loss.item()
            train_loss.append(loss.item())
            loss_dict = {'loss': train_loss[-1],
                         'iteration': self.current_iteration}
            # self.logger.add_scalar(
            #     f'TrainLoss', loss_dict['loss'], loss_dict['iteration'])
            self.logger.add_scalar(
                f'loss_mse', loss_mse, loss_dict['iteration'])
            self.logger.add_scalar(
                f'loss_contrastive', loss_contrastive, loss_dict['iteration'])
            self.logger.add_scalar(
                f'TrainLoss', loss_dict['loss'], loss_dict['iteration'])

            per_sample_loss = train_loss[-1] / self.config.batch_size
            per_sample_loss_mse = loss_mse / self.config.batch_size
            per_sample_loss_contrastive = loss_contrastive / self.config.batch_size
            print(
                f'Iteration {iteration} done with per loss {per_sample_loss:0.4f}.')
            print(
                f'Iteration {iteration} done with per_sample_loss_mse {per_sample_loss_mse:0.4f}.')
            print(
                f'Iteration {iteration} done with per_sample_loss_contrastive {per_sample_loss_contrastive:0.4f}.')

            if iteration % test_iteration == 0 or iteration == total_iterations:
                self.test_dict['test_srocc']['iter_no'].append(
                    self.current_iteration)

                # saves the model according to test frequency
                print("Saving model before testing")
                # added self.regressor here to debug
                # see error in detail at : https://chat.openai.com/share/3d7fc845-a3ee-4483-b6ca-4d72cb2fbc0b
                self.save_model(self.model, self.aggregator, self.regressor,
                                self.optim, best=False)
                self.test_during_train()  # saves the model again if it's best here

                # I am setting to train in test function but just in case :>
                self.model = self.internlm_weight_mode(
                    self.model, self.internlm_trainable_params, trainable=True)
                self.model.train()
                self.aggregator = self.weight_mode(
                    self.aggregator, trainable=True)
                self.aggregator.train()
                self.regressor = self.weight_mode(
                    self.regressor, trainable=True)
                self.regressor.train()

            self.current_iteration += 1

            del sampled_batch, dist_img_input, mos_target, hidden_states_dists, attention_output_dists, attention_output_refs, scores
            torch.cuda.empty_cache()

        return

    # The following function is to test in between
    def test_during_train(self):
        with torch.no_grad():
            self.model = self.internlm_weight_mode(
                self.model, self.internlm_trainable_params, trainable=False)
            self.model.eval()
            self.aggregator = self.weight_mode(
                self.aggregator, trainable=False)
            self.aggregator.eval()
            self.regressor = self.weight_mode(self.regressor, trainable=False)
            self.regressor.eval()

            self.test_dict['csv'] = {'Video_Name': [], 'MOS': [
            ], f'pred{self.current_iteration:04d}': []}

            test_prediction, corresponding_mos, corresponding_name = exp2_2_test_function(
                self.config, self.pooled_test_loader, self.model, self.aggregator, self.regressor, self.device)

            self.test_dict['csv'][f'pred{self.current_iteration:04}'] = test_prediction
            self.test_dict['csv']['Video_Name'] = corresponding_name
            self.test_dict['csv']['MOS'] = corresponding_mos

            srcc_test_correlation = spearmanr(
                np.array(test_prediction), np.array(corresponding_mos))[0]
            plcc_test_correlation = pearsonr(
                np.array(test_prediction), np.array(corresponding_mos))[0]

            self.test_dict['csv'][f'pred{self.current_iteration:04}'].append(
                srcc_test_correlation)
            self.test_dict['csv']['Video_Name'].append('SROCC')
            self.test_dict['csv']['MOS'].append(-1.0)

            del test_prediction, corresponding_mos, corresponding_name
            gc.collect()

            details_path = os.path.join(self.config.results_dir, 'details.txt')
            logging.basicConfig(filename=details_path,
                                filemode='a', level=logging.DEBUG, format='')

            print(
                f"SRCC for {self.current_iteration:04}, {srcc_test_correlation}")
            print(
                f"PLCC for {self.current_iteration:04}, {plcc_test_correlation}")
            logging.info(
                f"SRCC for {self.current_iteration:04}, {srcc_test_correlation}")
            logging.info(
                f"PLCC for {self.current_iteration:04}, {plcc_test_correlation}")

            # Saving test performance to disk
            if not os.path.exists((Path(self.config.results_dir) / 'Test').as_posix()):
                os.mkdir((Path(self.config.results_dir) / 'Test').as_posix())

            save_dir = (Path(self.config.results_dir) /
                        f'Test/predictions.csv').as_posix()

            if self.save_flag:
                df = pd.DataFrame.from_dict(self.test_dict['csv'])
                df.to_csv(save_dir, index=False)
            else:
                df1 = pd.read_csv(save_dir)
                df1[f'pred{self.current_iteration:04}'] = self.test_dict['csv'][f'pred{self.current_iteration:04}']
                df1.to_csv(save_dir, index=False)

            # So test_dict[test_srocc] looks like {[srocc1 srocc2] [iter 1 2]}
            self.test_dict['test_srocc']['srocc_value'].append(
                srcc_test_correlation)

            # Saving the test performance vs cycles
            pyplot.figure(1)
            pyplot.plot(self.test_dict['test_srocc']['iter_no'],
                        self.test_dict['test_srocc']['srocc_value'])
            pyplot.grid()
            pyplot.xlabel('Training Iteration')
            pyplot.ylabel('SROCC')
            pyplot.savefig(Path(self.config.results_dir) / f'Test/test.png')

            self.save_flag = False

            # Setting all network parameters to train() mode so that we can identify which trainable parameters to save
            self.model = self.internlm_weight_mode(
                self.model, self.internlm_trainable_params, trainable=True)
            self.model.train()
            self.aggregator = self.weight_mode(self.aggregator, trainable=True)
            self.aggregator.train()
            self.regressor = self.weight_mode(self.regressor, trainable=True)
            self.regressor.train()

            # Saving the model if it's the best model
            if len(self.test_srocc['srocc']) != 0:
                if srcc_test_correlation > max(self.test_srocc['srocc']):
                    self.save_model(self.model, self.aggregator,
                                    self.regressor, self.optim, best=True)

            self.test_srocc['srocc'].append(srcc_test_correlation)
            self.test_srocc['iteration'].append(self.current_iteration)

        return


def configuration_params():
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument('--img_dir', type=str,
                        default='/home/sanjotst/llm_iqa/llm-iqa/datasets/LIVE_FB/')
    parser.add_argument('--input_csv_file', type=str,
                        default='/home/sanjotst/llm_iqa/llm-iqa/labels/live_fb_split.csv')
    parser.add_argument('--synthetic_img_dir', type=str,
                        default='/home/sanjotst/llm_iqa/llm-iqa/datasets/live_fb_synthetic/LIVE_FB_synthetic_full_v2')

    # network args
    # or b_attn. Indicates basic attention or complex attention
    parser.add_argument('--network_type', type=str, default='b_attn')
    parser.add_argument('--embed_dim', type=int, default=4096)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--internlm_trainable_params',
                        type=list, default=['Qformer'])
    parser.add_argument('--optimizer_params', type=list, default=['Qformer'])
    parser.add_argument('--logit_processing_type',
                        type=str, default='init')  # or init

    # training args
    parser.add_argument('--mode', type=str, default="train")  # or eval
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--default_device', type=int, default=3)
    parser.add_argument('--num_gpus', type=int, default=2)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=16)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--test_epoch', type=float, default=0.5)
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--alpha_scaling', type=float,
                        default=100)  # for scaling the 2 losses
    # optimizer args
    parser.add_argument('--optim', type=str, default="adamw")
    parser.add_argument('--weight_decay', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=bool, default=False)
    parser.add_argument('--lr_scheduler', type=str, default='cosine_annealing')
    parser.add_argument('--cawr_restart_iter', default=200, type=int,
                        help='Restart at cosine annealig at the following itertion')
    parser.add_argument('--lwca_warmup_iter', default=1000, type=int,
                        help='Warmup iterations for linear warmup cosine annealing')

    # saving and resuming args
    parser.add_argument('--results_dir', type=str,
                        default='./exp2_triplet_loss_5k/')
    parser.add_argument('--annotation_directory', type=str,
                        default='/home/sanjotst/llm_iqa/llm-iqa/code/baselines/annotated_fsim_matrices')
    parser.add_argument('--resume_training', default=False,
                        type=bool, help='Resume training from a checkpoint')
    parser.add_argument('--resume_model_path', type=str,
                        default='/home/sanjotst/llm_iqa/llm-iqa/code/baselines/exp2_triplet_loss_5k/Run0002/Train/iter_2496.tar')
    parser.add_argument('--experiment description', type=str,
                        default='triplet loss with batch size 8 and custom 5k image dataset, to be benchmarked against PIPAL')
    config = parser.parse_args()
    return config


def main():
    config = configuration_params()
    model = DNNIter(config)
    model.train_model()

    return


if __name__ == '__main__':
    print('Program started at ' +
          datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = str(e)
    end_time = time.time()
    print('Program ended at ' +
          datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
