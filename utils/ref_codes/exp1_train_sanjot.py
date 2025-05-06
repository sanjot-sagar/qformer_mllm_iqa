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
from custom_losses.custom_blended_ordering_loss import CustomBlendedOrderingLoss


class DNNIter:
    def __init__(self):
        # self.exp_name = 'blending_loss_8_blends_temp_1_lr_e5_thresh_0.05'
        self.exp_name = 'test'
        self.results_folder = './results/'
        self.model_loc = join(self.results_folder, self.exp_name)
        os.makedirs(self.model_loc, exist_ok=True)

        self.random_seed = 0

        self.default_device = 6

        torch.cuda.set_device(self.default_device)
        self.device = f"cuda:{self.default_device}" if torch.cuda.is_available(
        ) else "cpu"

        self.num_gpus = 4

        self.percentage = 0.05
        self.batch_size = 8

        self.num_iterations = 1000
        self.iter_save_freq = 0
        self.max_iter = 0

        # blending params
        self.blend_levels = 8

        # Optimizer params
        # Optimizer learning rate
        self.lr = 1e-5
        # Optimizer (sgd/adam/adamw)
        self.optimizer = 'adamw'
        # Optimizer weight decay
        self.weight_decay = 0

        # Scheduler params
        # Optimizer learning rate schedule (none/cosine_annealing/cosine_annealing_warm_restart)
        self.lr_scheduler = 'none'
        # Restart at cosine annealig at the following itertion
        self.cawr_restart_iter = 200
        # Warmup iterations for linear warmup cosine annealing
        self.lwca_warmup_iter = 1000
        # Saving frequency per iteration (Do not save if 0)'
        self.iter_save_freq = 0

        # setting randomness
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def get_train_loader(self):
        img_dir = '/home/sanjotst/llm_iqa/internlm-sst/mnt_sanjot/LIVE_FB/'
        input_json = '/home/sanjotst/llm_iqa/IQA_outputs/mix-internlm-xcomposer-vl-hyperparameter-1/flive.json'

        df_data = pd.read_json(input_json) \
            .astype({'gt_score': np.float32, 'score': np.float32}) \
            .sort_values(by=['score', 'img_path'])
        n = int(len(df_data) * self.percentage)
        df_data_hq, df_data_lq = df_data[-n:], df_data[:n]
        train_transform = transforms.Compose([
            transforms.Resize((224, 224),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_data_hq = CustomDataset(img_dir, df_data_hq, train_transform)
        train_hq_loader = DataLoader(
            train_data_hq, batch_size=self.batch_size, shuffle=True, drop_last=True)
        train_data_lq = CustomDataset(img_dir, df_data_lq, train_transform)
        train_lq_loader = DataLoader(
            train_data_lq, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return train_hq_loader, train_lq_loader

    def get_model(self):
        # model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
        checkpoint = 'internlm/internlm-xcomposer-vl-7b'
        model = AutoModel.from_pretrained(
            checkpoint, trust_remote_code=True, torch_dtype=torch.float32).cuda().eval()
        print(f"the precision of the model is {model.internlm_model.dtype}")
        print("Initialized model.")
        device_map = auto_configure_device_map(
            self.num_gpus, self.default_device)
        print("Configured device_map.")
        print(device_map)
        model = dispatch_model(model, device_map=device_map)
        print("Dispatched model as per device_map.")
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True)
        model.tokenizer = tokenizer

        # freeze Layer: visual_encoder, Type: VisionTransformer
        model.visual_encoder.requires_grad_(False)
        # freeze Layer: ln_vision, Type: LayerNorm
        model.ln_vision.requires_grad_(False)
        # unfreeze Layer: Qformer, Type: BertLMHeadModel
        model.Qformer.requires_grad_(True)
        # freeze Layer: internlm_model, Type: InternLMForCausalLM
        # set these two to True because these gradients are required
        model.internlm_model.requires_grad_(True)
        # freeze Layer: internlm_proj, Type: Linear
        model.internlm_proj.requires_grad_(True)
        return model

    def get_optimizer(self, model):
        if model == None:
            return None
        # in parameters only use qformer , yaha pe sirf sampler/qformer ke parameters update karna hai
        param_list = [{'params': model.Qformer.parameters()}]

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

    def get_loss_fn(self):
        return CustomBlendedOrderingLoss()
    # use torch everywhere, with numpy gradients will not flow

    def softmax(self, a, b, temperature=1):
        input_tensor = torch.stack([a, b])
        input_tensor /= temperature

        # Compute softmax using PyTorch operations
        softmax_result = torch.nn.functional.softmax(input_tensor, dim=0)

        return softmax_result[0]

    def get_logits(self, model, images):
        text = "User: Rate the quality of the image. <ImageHere>" \
               "\nAssistant: The quality of the image is "

        with torch.cuda.amp.autocast():
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            img_embeds = model.encode_img(images)
            print(f"the datatype of img_embeds is {img_embeds.dtype}")
        prompt_segs = text.split('<ImageHere>')
        prompt_seg_tokens = [
            model.tokenizer(seg,  # segments are passed into the tokeniser
                            return_tensors='pt',  # return type is torch tensor
                            add_special_tokens=i == 0).  # special token added if the token is 1st
            # tensors moved to device and converted to int
            to(model.internlm_model.model.embed_tokens.weight.device).input_ids
            # Todo remove autocast to int above

            # loop over all segments in the list
            for i, seg in enumerate(prompt_segs)
        ]

        # list of tensors where each tensor represents embedding of tokenised segment
        prompt_seg_embs = [
            # interlm_model attribute has model attribute
            model.internlm_model.model.embed_tokens(seg).expand(
                img_embeds.shape[0], -1, -1)  # all prompt segment embeds so these are also
            for seg in prompt_seg_tokens
        ]
        # print(len((prompt_seg_embs)))  # Check its length and elements
        # try shape here
        # List of embeddings, containing first segment, image emedding, containing embedding of 2nd prompt segment
        # what is the third term
        # 0 -> 1*10*4096, 1 -> 1*9*4096, 1 image embed -> 1*66*4096,
        print(
            f"the datatype of prompt_seg_embs[0] is {prompt_seg_embs[0].dtype}")
        prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
        # dim = 1 means these embeddings are not 1d, what is the dimension of these embeddings
        prompt_embs = torch.cat(prompt_seg_embs, dim=1)
        print(f"the datatype of prompt_embs is {prompt_embs.dtype}")
        # access internlm_model attribute of the model, send prompt_embed as inputs
        # logits typically represent the unnormalized output scores before applying a softmax function
        # Logits are the raw pre-softmax scores for each token.
        # all rows of last column are selected
        # it often represents the model's prediction for a specific task or prompt.
        # 48
        # x = model.internlm_model(
        #     inputs_embeds=prompt_embs).get_encoder().encoder.layers[-1].output
        outputs = model.internlm_model(
            inputs_embeds=prompt_embs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        print(logits.shape)
        # print("data type of x")
        # print(x.dtype) #AttributeError: 'NoneType' object has no attribute 'dtype'
        print("the hidden states are")
        print(hidden_states)
        print("the shape of  hidden states is")

        print(hidden_states.shape)

        exit()
        # print(x.shape)
        # should not have item, gradients will not pass
        # shape : 48 * 1, remove clone
        lgood, lpoor = x[:, 18682], x[:, 5527]
        print(f"the datatype of lgood is {lgood.dtype}")
        return lgood, lpoor  # after vectorisation each of them should be 48*1

    def evaluate_score(self, img):
        # get_logits doesn't take a batch of images as the model
        # processes 1 image at a time inside get_logits function
        logits_good, logits_bad = self.get_logits(model, img)
        score = self.softmax(logits_good, logits_bad)  # 48 * 1
        print(
            f"the value of logits_good is {logits_good}. the value of logits_bad is {logits_bad}")
        return score

    def train_model(self, train_hq_loader, train_lq_loader,  model, optimizer, scheduler):

        writer = SummaryWriter(log_dir=join(
            self.results_folder, 'runs', self.exp_name))
        # model = model.to(self.device)
        model.train()
        train_hq_loader_iter = iter(train_hq_loader)
        train_lq_loader_iter = iter(train_lq_loader)

        for iteration in tqdm(range(1, self.num_iterations+1)):
            if self.max_iter and iteration >= self.max_iter:
                break
            # get image batch
            # in case of uneven number of lq and hq images
            try:
                x_hq_batch, x_hq_score_batch = next(train_hq_loader_iter)
            except StopIteration:
                train_hq_loader_iter = iter(train_hq_loader)
                x_hq_batch, x_hq_score_batch = next(train_hq_loader_iter)

            try:
                x_lq_batch, x_lq_score_batch = next(train_lq_loader_iter)
            except StopIteration:
                train_lq_loader_iter = iter(train_lq_loader)
                x_lq_batch, x_lq_score_batch = next(train_lq_loader_iter)
            # print(x_hq_batch.shape) # 16 x 3 x 224 x 224
            # load batches to device(cuda)
            x_hq_batch = x_hq_batch.to(self.device)
            x_lq_batch = x_lq_batch.to(self.device)
            optimizer.zero_grad()
            # blend
            # initialize blend_vec
            blend_vec = torch.linspace(
                0, 1, self.blend_levels, device=self.device
            ).view(1, 1, 1, 1, self.blend_levels)
            # Todo(Sanjot): verify if ordering is maintained?
            blends = blend_vec * x_lq_batch.unsqueeze(-1) \
                + (1-blend_vec) * x_hq_batch.unsqueeze(-1)
            # print(blends.shape)  # blends => bbs x 3 x H x W x bl => 16 x 3 x 224 x 224 x 3

            blends = torch.permute(blends, (0, 4, 1, 2, 3))
            # print(blends.shape)  # blends => bbs x bl x 3 x H x W => 16 x 3 x 3 x 224 x 224
            # Todo check blends value, replace with view and report error
            blends = blends.reshape(
                self.batch_size*self.blend_levels, blends.shape[2], blends.shape[3], blends.shape[4])
            # check if below operations can be done on gpu instead
            # search for what error was there here when a batch was sent
            # if model supports taking input of images as batch
            # every reshape to view, view is memory efficient
            scores = self.evaluate_score(blends).reshape(
                self.batch_size, self.blend_levels)
            print(scores)  # 16 * 3
            # do outside epoch and loss because it is called only once within epoch/iteration
            loss_fn = self.get_loss_fn()
            loss = loss_fn(scores)

            # Plots learning rate
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                break
            writer.add_scalar('extra_info/LR', lr, iteration)

            loss.backward()
            optimizer.step()
            # scheduler.step()

            writer.add_scalar('loss ', loss.item(), iteration)

            if self.iter_save_freq:
                if iteration % self.iter_save_freq == 1000:
                    torch.save(model.state_dict(), join(
                        self.model_loc, 'model_iter_%06d.pth' % (iteration)))

        print('training completed')
        writer.close()

        # uncomment to write model to disk
        save_loc = join(self.model_loc, 'model')
        os.makedirs(save_loc, exist_ok=True)
        model.save_pretrained(save_loc)
        print(f'model saved at path {save_loc}')


if __name__ == '__main__':
    dnn_iter = DNNIter()
    train_hq_loader, train_lq_loader = dnn_iter.get_train_loader()
    model = dnn_iter.get_model()
    # print(model)
    optimizer = dnn_iter.get_optimizer(model)
    scheduler = dnn_iter.get_scheduler(optimizer)
    dnn_iter.train_model(train_hq_loader, train_lq_loader,
                         model, optimizer, scheduler)
