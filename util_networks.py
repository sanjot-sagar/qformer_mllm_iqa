
# General Imports
import torch.nn as nn
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

# InternLM and QInstruct imports
from transformers import AutoModel, AutoTokenizer
from accelerate import dispatch_model
from transformers import AutoTokenizer
from utils.utils import auto_configure_device_map

# imports for quantisation
import torch
from transformers import AutoModel, AutoTokenizer


# InternLMv2 model device map import
from utils.device_map_internlm_v2 import auto_configure_device_map_v2


def get_internLM_model(config):
    checkpoint = 'internlm/internlm-xcomposer-vl-7b'
    model = AutoModel.from_pretrained(
        checkpoint, trust_remote_code=True, torch_dtype=torch.float32).cuda().eval()
    device_map = auto_configure_device_map(
        config.num_gpus, config.default_device)
    print("Configured device_map.")
    print(config.default_device)
    model = dispatch_model(model, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True)
    model.tokenizer = tokenizer
    return model

# quantised internlm model class

# from auto_gptq.modeling._base import BaseGPTQForCausalLM
# import auto_gptq
# auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]


# class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
#     layers_block_name = "model.layers"
#     outside_layer_modules = [
#         'vit', 'vision_proj', 'model.tok_embeddings', 'model.norm', 'output',
#     ]
#     inside_layer_modules = [
#         ["attention.wqkv.linear"],
#         ["attention.wo.linear"],
#         ["feed_forward.w1.linear", "feed_forward.w3.linear"],
#         ["feed_forward.w2.linear"],
#     ]

# # quantised internlm model
# # why is this model class automatically using mutliple GPUs


# def get_internLM_quantised_model(config):

#     checkpoint = 'internlm/internlm-xcomposer2-vl-7b-4bit'
#     model = InternLMXComposer2QForCausalLM.from_quantized(
#         checkpoint, trust_remote_code=True).eval()
#     model.to(config.default_device)
#     tokenizer = AutoTokenizer.from_pretrained(
#         checkpoint, trust_remote_code=True)
#     model.tokenizer = tokenizer
#     return model


# internlm model2
def get_internLM_v2_model(config):
    checkpoint = 'internlm/internlm-xcomposer2-vl-7b'
    model = AutoModel.from_pretrained(
        checkpoint, trust_remote_code=True, torch_dtype=torch.float32).cuda().eval()
    device_map = auto_configure_device_map_v2(
        config.num_gpus, config.default_device)
    print("Configured device_map.")
    model = dispatch_model(model, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True)
    model.tokenizer = tokenizer
    return model


# llava model
def get_llava_model(config):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    checkpoint = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(checkpoint)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=checkpoint,
        model_base=None,
        model_name=model_name,
        use_flash_attn=False

    )

    return model, tokenizer, image_processor

# mplug model


def get_mplug_owl_model(config):
    from mplug_owl2.model.builder import load_pretrained_model
    from mplug_owl2.mm_utils import get_model_name_from_path
    checkpoint = "MAGAer13/mplug-owl2-llama2-7b"
    model_name = get_model_name_from_path(checkpoint)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=checkpoint,
        model_base=None,
        model_name=model_name,
        device_map='auto'
    )

    return model, tokenizer, image_processor

# qinstruct internlm model


def get_qinstruct_internLM_model(config):
    checkpoint = 'DLight1551/internlm-xcomposer-vl-7b-qinstruct-full'
    model = AutoModel.from_pretrained(
        checkpoint, trust_remote_code=True, torch_dtype=torch.float32).cuda().eval()
    device_map = auto_configure_device_map(
        config.num_gpus, config.default_device)
    print("Configured device_map.")
    model = dispatch_model(model, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True)
    model.tokenizer = tokenizer
    return model


# Takes the bs*78*4096 tensor which indicates (batch_size, sequence_length, embedding_size) and returns a bs*1 tensor
# Or takes bs*4096 tensor and returns a bs*1 tensor
class NormalRegressor(nn.Module):
    def __init__(self, embed_dim, pool=False):
        super().__init__()
        self.pool = pool
        if self.pool:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)

        hidden_layer_dim = embed_dim // 2
        self.fc1 = nn.Linear(embed_dim, hidden_layer_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_dim, 1)

    def forward(self, x):
        # x is of shape (batch_size, sequence_length, embed_dim)
        # Permute x to shape (batch_size, embed_dim, sequence_length) for adaptive avg pooling
        # Add this line to check input tensor shape
        if self.pool:
            # Print shape before permute operation
            x = x.permute(0, 2, 1)
            x = self.avg_pool(x)
            x = x.squeeze(2)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
#########################################################################
# NormalRegressor1
# Takes the bs*78*4096 tensor which indicates (batch_size, sequence_length, embedding_size) and returns a bs*1 tensor
# Or takes bs*4096 tensor and returns a bs*1 tensor


class NormalRegressor1(nn.Module):
    def __init__(self, embed_dim, pool=False):
        super().__init__()
        self.pool = pool
        if self.pool:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x is of shape (batch_size, sequence_length, embed_dim)
        # Permute x to shape (batch_size, embed_dim, sequence_length) for adaptive avg pooling
        # Add this line to check input tensor shape
        if self.pool:
            # Print shape before permute operation
            print("Shape of x before permute:", x.shape)
            x = x.permute(0, 2, 1)
            x = self.avg_pool(x)
            x = x.squeeze(2)

        x = self.fc1(x)
        # print("normal regressor 1 is being used ")
        return x
##############################################################################################################

# Nithin's Self-Attention network


class AttentionAggregator(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_layers=8, add_mlp=True, add_ln=True, residual=True):
        super().__init__()
        self.num_layers = num_layers
        self.add_mlp = add_mlp
        self.add_ln = add_ln
        self.residual = residual
        self.self_attention_aggregators = nn.ModuleList([nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True) for _ in range(num_layers)])
        if add_mlp:
            self.mlps = nn.ModuleList([MLP([embed_dim, embed_dim], mlp_layers=[
                                      'linear', 'dropout'], drp_p=0.1) for _ in range(num_layers)])
        if add_ln:
            self.lns = nn.ModuleList([nn.LayerNorm(embed_dim)
                                     for _ in range(num_layers)])
        self.cls_token = nn.Parameter(torch.randn(
            1, 1, embed_dim), requires_grad=True)

        # Add a linear layer that maps from embed_dim to 1
        self.final_output_layer = nn.Linear(embed_dim, 1)

    def forward(self, x):   # B x L x F
        # B x (L+1) x F
        # x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)

        attn_weights_list = []
        hidden_layers = []
        for i in range(self.num_layers):
            x_out, attn_weights = self.self_attention_aggregators[i](
                x, x, x, need_weights=True, average_attn_weights=False)
            attn_weights_list.append(attn_weights)
            if self.add_mlp:
                x_out = self.mlps[i](x_out)
            x = x + x_out if self.residual else x_out
            if self.add_ln:
                # LayerNorm after adding residual - similar to Bert
                x = self.lns[i](x)
            hidden_layers.append(x)

            output = x[:, :1, :]   # B x F
            # print(output.shape)
            final_output = self.final_output_layer(output)

        # return {'output': final_output,
        #         'hidden_layers': hidden_layers,         # List of B x (L+1) x F
        #         # List of B x H x (L+1) x (L+1)
        #         'attn_weights': attn_weights_list,
        #         }
        return final_output


class MLP(nn.Module):
    def __init__(self, dimension_list, mlp_layers=['linear', 'bn', 'relu', 'dropout', 'linear'], drp_p=0.1):
        super().__init__()
        self.drp_p = drp_p
        layer_list = []
        dimension_pair_list = list(
            zip(dimension_list, dimension_list[1:]+[dimension_list[-1]]))
        j = 0
        for i, layer in enumerate(mlp_layers):
            if layer == 'linear':
                layer_list.append(self.load_layer(
                    layer, dimension_pair_list[j]))
                j += 1
            elif layer in ['bn', 'ln']:
                layer_list.append(self.load_layer(
                    layer, [dimension_pair_list[j][0]]))
            else:
                layer_list.append(self.load_layer(layer))
        self.out_network = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.out_network(x)

    def load_layer(self, layer_name, dimensions=None, args=None):
        if layer_name == 'linear':
            return nn.Linear(in_features=dimensions[0], out_features=dimensions[1])
        elif layer_name == 'dropout':
            return nn.Dropout(self.drp_p)
        elif layer_name == 'bn':
            return nn.BatchNorm1d(num_features=dimensions[0])
        elif layer_name == 'ln':
            return nn.LayerNorm(dimensions[0])
        elif layer_name == 'gelu':
            return nn.GELU()
        elif layer_name == 'relu':
            return nn.ReLU()


class BasicMultiheadAttentionAggregator(nn.Module):
    def __init__(self, embed_dim, num_heads, regressor_bool=True):
        super(BasicMultiheadAttentionAggregator, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads)

        self.regressor_bool = regressor_bool
        if regressor_bool:
            self.regressor = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x shape is (batch_size, seq_length, embed_dim)
        # MultiheadAttention expects input in the shape of (seq_length, batch_size, embed_dim)
        # print(x.shape)
        x = x.permute(1, 0, 2)
        # print("shape after permutation")
        # print(x.shape)
        x = x.to('cuda')
        x, _ = self.attention(x, x, x)

        # Aggregate across the sequence length (e.g., take the mean)
        x = x.mean(dim=0)  # shape = (batch_size, embed_dim)
        if self.regressor_bool:
            x = self.regressor(x)  # shape = (batch_size, 1)
        return x


class BasicMultiheadAttentionAggregator1(nn.Module):
    def __init__(self, embed_dim, num_heads, regressor_bool=True):
        super(BasicMultiheadAttentionAggregator1, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads)

        self.regressor_bool = regressor_bool
        if regressor_bool:
            hidden_layer_dim = embed_dim // 2
            self.fc1 = nn.Linear(embed_dim, hidden_layer_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_layer_dim, 1)

    def forward(self, x):
        # x shape is (batch_size, seq_length, embed_dim)
        # MultiheadAttention expects input in the shape of (seq_length, batch_size, embed_dim)
        x = x.permute(1, 0, 2)
        x, _ = self.attention(x, x, x)

        # Aggregate across the sequence length (e.g., take the mean)
        x = x.mean(dim=0)  # shape = (batch_size, embed_dim)

        if self.regressor_bool:
            # Apply the defined layers only if regressor_bool is True
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)  # shape = (batch_size, 1)

        print(f"using fc1 relu fc2")
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.layernorm2(x + ff_output)
        return x


class ComplexMultiheadAttentionAggregator(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=1, ff_dim=2048, regressor_bool=True):
        super(ComplexMultiheadAttentionAggregator, self).__init__()

        self.regressor_bool = regressor_bool
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])

        if regressor_bool:
            self.regressor = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x shape is (batch_size, seq_length, embed_dim)
        # MultiheadAttention expects input in the shape of (seq_length, batch_size, embed_dim)
        x = x.permute(1, 0, 2)

        for block in self.transformer_blocks:
            x = block(x)

        # Aggregate across the sequence length (e.g., take the mean)
        x = x.mean(dim=0)  # shape = (batch_size, embed_dim)
        if self.regressor_bool:
            print("using complex aggregator's regressor")
            x = self.regressor(x)  # shape = (batch_size, 1)
        return x

class no_qformer_aggregator(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=12, regressor_bool=True):
        super(no_qformer_aggregator, self).__init__()
        self.num_layers = num_layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads) for _ in range(num_layers)
        ])

        self.regressor_bool = regressor_bool
        if regressor_bool:
            self.regressor = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x shape is (batch_size, seq_length, embed_dim)
        # MultiheadAttention expects input in the shape of (seq_length, batch_size, embed_dim)
        # print(x.shape) # torch.Size([32, 257, 1408])
        x = x.permute(1, 0, 2)  # Change to (seq_length, batch_size, embed_dim)
        x = x.to('cuda')

        # Apply self-attention over 12 layers
        for attention in self.attention_layers:
            x, _ = attention(x, x, x)

        # Aggregate across the sequence length (e.g., take the mean)
        x = x.mean(dim=0)  # shape = (batch_size, embed_dim)

        if self.regressor_bool:
            x = self.regressor(x)  # shape = (batch_size, 1)

        return x
