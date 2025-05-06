import argparse
import json
import math
import os
import random
from itertools import cycle
from os.path import join

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from accelerate import dispatch_model
from transformers import AutoTokenizer
from utils.utils import auto_configure_device_map
from custom_datasets.llm_dataset import CustomDataset
from torch.nn import MultiheadAttention

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

# action point 1 : replace this part with nithin's newer code
# https://github.com/nithincbabu7/pytorch-training-templates/blob/main/graphs/models/custom_models.py


class AttentionAggregator(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.self_attention_aggregator = MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.cls_token = nn.Parameter(torch.randn(
            1, 1, embed_dim), requires_grad=True)
        # Add a linear layer that maps from embed_dim to 1
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, x):
        print(
            f"x device: {x.device}, cls_token device: {self.cls_token.device}")
        # cls_token = self.cls_token.to(x.device), why does this not work
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        x_out, attn_weights = self.self_attention_aggregator(
            x, x, x, need_weights=True, average_attn_weights=False)
        # Use the transformed cls_token (first element of the sequence) for classification
        cls_output = x_out[:, 0, :]  # Shape: (B, E), check shape here
        # Pass the cls_output through the linear layer
        final_output = self.output_layer(cls_output)  # Shape: (B, 1)
        print("the shape of the final output is")
        print(final_output.shape)  # batch_size * 1
        return {'output': final_output,
                'x_out': x_out,
                'attn_weights': attn_weights,
                }


class DNNIter:
    def __init__(self):
        self.exp_name = 'exp3_qformer_attention_aggregator'
        self.results_folder = './results/'
        self.model_loc = join(self.results_folder, self.exp_name)
        os.makedirs(self.model_loc, exist_ok=True)
        self.random_seed = 0
        self.default_device = 6
        torch.cuda.set_device(self.default_device)
        self.device = f"cuda:{self.default_device}" if torch.cuda.is_available(
        ) else "cpu"
        self.num_gpus = 4
        self.batch_size = 64
        self.num_iterations = 625
        self.iter_save_freq = 0
        self.max_iter = 0
        self.lr = 1e-4
        self.optimizer = 'adamw'
        self.weight_decay = 0
        self.lr_scheduler = 'none'
        self.iter_save_freq = 0
        self.aggregator = AttentionAggregator(
            embed_dim=4096).to(self.device)  # Adjust embed_dim as necessary
        # setting randomness
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def get_train_loader(self):
        img_dir = '/home/sanjotst/llm_iqa/internlm-sst/mnt_sanjot/LIVE_FB/'
        input_json = '/home/sanjotst/llm_iqa/sanjot_json/flive.json'

        df_data = pd.read_json(input_json) \
            .astype({'gt_score': np.float32})
        train_transform = transforms.Compose([
            transforms.Resize((224, 224),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_data = CustomDataset(img_dir, df_data, train_transform)
        # should drop last be true here ?
        train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return train_loader

    def get_model(self):
        checkpoint = 'internlm/internlm-xcomposer-vl-7b'
        model = AutoModel.from_pretrained(
            checkpoint, trust_remote_code=True, torch_dtype=torch.float32).cuda().eval()
        device_map = auto_configure_device_map(
            self.num_gpus, self.default_device)
        print("Configured device_map.")
        model = dispatch_model(model, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True)
        model.tokenizer = tokenizer
        model.visual_encoder.requires_grad_(False)
        model.ln_vision.requires_grad_(False)
        model.Qformer.requires_grad_(True)
        model.internlm_model.requires_grad_(True)
        model.internlm_proj.requires_grad_(True)
        return model

    def get_optimizer(self, aggregator, model):
        if model == None:
            return None
        # use dictionaries instead this code may not be correct
        param_list = [{'params': list(
            aggregator.parameters()) + list(model.Qformer.parameters())}]
        if self.optimizer == 'sgd':
            return torch.optim.SGD(
                param_list, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer == 'adam':
            return torch.optim.Adam(
                param_list, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer == 'adamw':
            return torch.optim.AdamW(
                param_list, lr=self.lr, weight_decay=self.weight_decay)
        return None

    def get_scheduler(self, optimizer):
        if optimizer == None:
            return None
        if self.lr_scheduler == 'cosine_annealing':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.num_iterations)
        if self.lr_scheduler == 'cosine_annealing_warm_restart':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.cawr_restart_iter)
        if self.lr_scheduler == 'linear_warmup_cosine_annealing':
            lr_lambda = (
                lambda cur_iter: cur_iter / self.lwca_warmup_iter
                if cur_iter <= self.lwca_warmup_iter
                else 0.5 * (1 + math.cos(math.pi * (cur_iter - self.lwca_warmup_iter) / (self.num_iterations - self.lwca_warmup_iter)))
            )
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda)
        return None

    def get_loss_fn(self, predictions, ground_truth):
        print(predictions.shape)
        # Remove the extra dimension from predictions
        predictions = predictions.squeeze(-1)  # batch_size * 1 => batch_size
        return torch.nn.functional.mse_loss(predictions, ground_truth)

    def get_logits(self, model, images):
        # report the int() usage because period is before image
        text = "User: Rate the quality of the image. <ImageHere>" \

        with torch.cuda.amp.autocast():
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            img_embeds = model.encode_img(images)
        prompt_segs = text.split('<ImageHere>')
        prompt_seg_tokens = [
            model.tokenizer(seg,
                            return_tensors='pt',
                            add_special_tokens=i == 0).
            to(model.internlm_model.model.embed_tokens.weight.device).input_ids
            for i, seg in enumerate(prompt_segs)
        ]
        prompt_seg_embs = [
            model.internlm_model.model.embed_tokens(seg).expand(
                img_embeds.shape[0], -1, -1)
            for seg in prompt_seg_tokens
        ]
        prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
        prompt_embs = torch.cat(prompt_seg_embs, dim=1)
        outputs = model.internlm_model(
            inputs_embeds=prompt_embs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        # hidden state :  33 tensor tuple
        # hidden states [-1] shape 64*88*4096 -> input to trainable attention class module
        # apply self attention to aggregate to get 64*4096 vector
        return hidden_states

    def evaluate_score(self, hidden_states):
        scores_dict = self.aggregator(hidden_states)
        # print(scores_dict['output'].shape) # torch.Size([batch_size, 1])
        return scores_dict['output']

    def train_model(self, train_loader,  model, optimizer, scheduler):
        # convert to epoch template : https://github.com/nithincbabu7/pytorch-training-templates/blob/main/train_epoch.py
        writer = SummaryWriter(log_dir=join(
            self.results_folder, 'runs', self.exp_name))
        model.train()
        train_loader_iter = iter(train_loader)
        for iteration in tqdm(range(1, self.num_iterations+1)):
            if self.max_iter and iteration >= self.max_iter:
                break
            try:
                x_batch, x_score_batch = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                x_batch, x_score_batch = next(train_loader_iter)
            # print(x_batch.shape)  # batch_size x 3 x 224 x 224
            # load batches to device(cuda)
            # print(x_score_batch.shape) # torch.Size([batch_size])
            # print(x_score_batch)
            x_batch = x_batch.to(self.device)
            optimizer.zero_grad()
            # calculate scores and use them to calculate loss
            hidden_states = self.get_logits(model, x_batch)
            # qinstruct code :
            scores = self.evaluate_score(hidden_states)
            print(scores)
            # do outside epoch and loss because it is called only once within epoch/iteration, why ?
            # Ensure ground truth is on the same device as predictions
            x_score_batch = x_score_batch.to(self.device)
            loss = self.get_loss_fn(scores, x_score_batch)
            # Plots learning rate
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                break
            writer.add_scalar('extra_info/LR', lr, iteration)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            writer.add_scalar('loss ', loss.item(), iteration)

        print('training completed')
        writer.close()
        save_loc = join(self.model_loc, 'model')
        os.makedirs(save_loc, exist_ok=True)
        model.save_pretrained(save_loc)
        print(f'model saved at path {save_loc}')
        aggregator_state_dict = self.aggregator.state_dict()
        torch.save(aggregator_state_dict, 'multihead_attention_state_dict.pth')


if __name__ == '__main__':
    dnn_iter = DNNIter()
    train_loader = dnn_iter.get_train_loader()
    model = dnn_iter.get_model()
    optimizer = dnn_iter.get_optimizer(dnn_iter.aggregator, model)
    scheduler = dnn_iter.get_scheduler(optimizer)
    dnn_iter.train_model(train_loader,
                         model, optimizer, scheduler)