import argparse
import json
import math
import os
import random
from itertools import cycle
from os.path import join

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
from custom_datasets.llm_dataset import CustomDataset
from scipy.stats import spearmanr, pearsonr
from torch.nn import MultiheadAttention


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
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        x_out, attn_weights = self.self_attention_aggregator(
            x, x, x, need_weights=True, average_attn_weights=False)
        cls_output = x_out[:, 0, :]
        final_output = self.output_layer(cls_output)
        return {'output': final_output,
                'x_out': x_out,
                'attn_weights': attn_weights}


class DNNIter:
    def __init__(self):
        self.exp_name = 'inference'
        self.random_seed = 0
        self.default_device = 6
        self.batch_size = 128
        torch.cuda.set_device(self.default_device)
        self.device = f"cuda:{self.default_device}" if torch.cuda.is_available(
        ) else "cpu"
        # Initialize and load the AttentionAggregator
        self.aggregator = AttentionAggregator(embed_dim=4096, num_heads=8).to(
            self.device)  # Adjust as per your model
        aggregator_state_dict = torch.load(
            'multihead_attention_state_dict.pth')
        self.aggregator.load_state_dict(aggregator_state_dict)

        torch.cuda.set_device(self.default_device)
        self.device = f"cuda:{self.default_device}" if torch.cuda.is_available(
        ) else "cpu"

        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def get_train_loader(self):
        img_dir = '/home/sanjotst/llm_iqa/internlm-sst/mnt_sanjot/LIVE_FB'
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
        train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=False)
        return train_loader

    def get_model(self):
        checkpoint = 'internlm/internlm-xcomposer-vl-7b'
        model = AutoModel.from_pretrained(
            checkpoint, trust_remote_code=True).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(
            'internlm/internlm-xcomposer-vl-7b', trust_remote_code=True)
        model.tokenizer = tokenizer
        return model

    def get_logits(self, model, images):
        # report the int() usage because period is before image
        text = "User: Rate the quality of the image. <ImageHere>" \
            # int assumption, please read qinstruct code also
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

    def inference_model(self, train_loader, model):
        # Load original JSON data
        input_json = '/home/sanjotst/llm_iqa/sanjot_json/flive.json'

        with open(input_json, 'r') as f:
            iqa_data = json.load(f)
        all_predictions = []
        all_gt_scores = []
        iqa_index = 0  # Initialize index to keep track of the current entry in iqa_data

        with torch.no_grad():
            for images, gt_scores in tqdm(train_loader, desc="Inference"):
                images = images.to(self.device)
                # Perform inference on the entire batch
                hidden_states = self.get_logits(model, images)
                predicted_scores = self.evaluate_score(hidden_states)
                # Assuming predicted_scores is a tensor that needs squeezing
                for score in predicted_scores.squeeze().tolist():
                    iqa_data[iqa_index]["predicted_score"] = score
                    iqa_index += 1
                # Collect predictions for each image
                batch_predictions = predicted_scores.tolist()
                all_predictions.extend(batch_predictions)
                all_gt_scores.extend(gt_scores.tolist())
        # Calculate SROCC and PLCC
        srocc = spearmanr(all_gt_scores, all_predictions).correlation
        # plcc = pearsonr(all_gt_scores, all_predictions)[0]

        print("SROCC:", srocc)
        # print("PLCC:", plcc)

        # Save modified iqa_data with predictions to a new JSON file
        output_json = 'IQA_outputs/predictions_with_scores_test.json'
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as wf:
            json.dump(iqa_data, wf, indent=4)


if __name__ == '__main__':
    dnn_iter = DNNIter()
    train_loader = dnn_iter.get_train_loader()
    model = dnn_iter.get_model()
    dnn_iter.inference_model(train_loader, model)
