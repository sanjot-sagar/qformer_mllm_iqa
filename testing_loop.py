
import time
import datetime
import traceback
import argparse
import numpy as np
import json
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
import itertools
import csv

# File imports
from util_networks import *
from util_dataload import *
from llm_feature_extraction import *


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
    return


def load_model( model, aggregator, path):
    checkpoint = torch.load(path, map_location=default_device)
    model.load_state_dict(checkpoint['model']['state_dict'], strict=False)
    aggregator.load_state_dict(checkpoint['aggregator']['state_dict'])
    if config.model_type != 'no_qformer':
        regressor.load_state_dict(checkpoint['regressor']['state_dict'])
        return model, aggregator, regressor
    else :
        return model, aggregator

# this is a legacy function, not sure if needed


def weight_mode(model, trainable=False):
    for param in model.parameters():
        if trainable:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    return model


def init_test_dataloaders(config):
    test_data = get_test_data(config)
    processed_test_data = CustomDataset(
        config.img_dir, test_data, config.model_description, data_transform='normal_transform')
    pooled_test_loader = DataLoader(
        processed_test_data, batch_size=8, shuffle=False)
    return pooled_test_loader


def get_predictions(config, test_loader, model, aggregator, regressor, device, tokenizer=None, image_processor=None):
    predicted_scores = []
    corresponding_name = []
    corresponding_mos = []
    default_device = config.default_device
    for sampled_batch in tqdm(test_loader):
        img_input = sampled_batch['img'].to(device)
        mos = sampled_batch['mos'].to(device)
        name = sampled_batch['name']

        if config.model_description == 'internlm_vl':
            if config.logit_processing_type == 'init':
                if config.model_type == 'no_qformer':
                    hidden_states = get_init_logits_no_qformer(model, img_input)
                    # hidden_states2 = get_init_logits_no_qformer(self.model, img_input2)
                else : 
                    hidden_states = get_init_logits(
                    model, img_input)
                # hidden_states = get_init_logits(
                #     model, img_input)
            elif config.logit_processing_type == 'sentence':
                hidden_states = get_sentence_logits(
                    model, img_input, gen_config)
        elif config.model_description == 'internlm_vl2':
            if config.logit_processing_type == 'init':
                hidden_states = get_init_logits_internlm_v2(
                    model, img_input)
            elif config.logit_processing_type == 'sentence':
                hidden_states = get_sentence_logits(
                    model, img_input, gen_config)
        elif config.model_description == 'internlm_vl2_quantised':
            if config.logit_processing_type == 'init':
                hidden_states = get_init_logits_internlm_quantised(
                    model, img_input, default_device)
            elif config.logit_processing_type == 'sentence':
                hidden_states = get_sentence_logits(
                    model, img_input, gen_config)
        elif config.model_description == 'llava':
            if config.logit_processing_type == 'init':
                hidden_states = get_init_logits_llava(
                    model, img_input, tokenizer, image_processor)
            elif config.logit_processing_type == 'sentence':
                hidden_states = get_sentence_logits(
                    model, img_input, gen_config)
        elif config.model_description == 'mplug_owl':
            if config.logit_processing_type == 'init':
                hidden_states = get_init_logits_mplug_owl(
                    model, img_input, tokenizer, image_processor)
            elif config.logit_processing_type == 'sentence':
                hidden_states = get_sentence_logits(
                    model, img_input, gen_config)

        predicted_video_feats = aggregator(hidden_states)
        if config.model_type == 'no_qformer':
            predicted_video_scores = predicted_video_feats
        else : 
            predicted_video_scores = regressor(predicted_video_feats)

        predicted_scores.append(predicted_video_scores.detach().cpu())
        corresponding_name.append(name)
        corresponding_mos.append(mos.detach().cpu())

        del sampled_batch, img_input, mos, hidden_states, predicted_video_scores
        torch.cuda.empty_cache()

    predicted_scores = torch.cat(
        predicted_scores, dim=0).squeeze().numpy().tolist()
    corresponding_mos = torch.cat(corresponding_mos, dim=0).squeeze().tolist()
    print(len(predicted_scores))
    print(len(corresponding_mos))

    try:
        corresponding_name = list(np.concatenate(corresponding_name))
    except:
        corresponding_name = corresponding_name

    return predicted_scores, corresponding_mos, corresponding_name


def save_predictions_to_csv(predicted_scores, corresponding_mos, corresponding_name, dataset_name, results_dir):
    csv_file_path = os.path.join(
        results_dir, f"{dataset_name}_predictions.csv")
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["Image Name", "MOS (Ground Truth)", "Predicted Value"])
        for name, mos, predicted in zip(corresponding_name, corresponding_mos, predicted_scores):
            writer.writerow([name, mos, predicted])
    print(f"CSV file for {dataset_name} saved at {csv_file_path}")


def test_model(config):
    # Set device
    torch.cuda.set_device(config.default_device)
    device = f"cuda:{config.default_device}" if torch.cuda.is_available(
    ) else "cpu"

    tokenizer = None
    image_processor = None

    # Get model
    if config.model_description == 'internlm_vl':
        model = get_internLM_model(config)
    if config.model_description == 'internlm_vl2_quantised':
        model = get_internLM_quantised_model(config)
    if config.model_description == 'internlm_vl2':
        model = get_internLM_v2_model(config)
    if config.model_description == 'llava':
        model, tokenizer, image_processor = get_llava_model(config)
    if config.model_description == 'mplug_owl':
        model, tokenizer, image_processor = get_mplug_owl_model(config)

    regressor = NormalRegressor1(config.embed_dim)

    if config.network_type == "b_attn":
        if config.model_type == 'no_qformer':
            aggregator = no_qformer_aggregator(
                1408, config.num_mha_heads, config.num_layers).to(device)
        else : 
            aggregator = BasicMultiheadAttentionAggregator(
                config.embed_dim, config.num_heads, regressor_bool=False)
    elif config.network_type == "attn":
        aggregator = AttentionAggregator(
            config.embed_dim, config.num_heads, regressor_bool=False)
    elif config.network_type == "c_attn":
        aggregator = ComplexMultiheadAttentionAggregator(
            config.embed_dim, config.num_heads)

    # Load model, aggregator, regressor
    # add model path here
    model_path = config.model_path
    print(model_path)
    if config.model_type == 'no_qformer':
        model, aggregator = load_model(
            config, model, aggregator, model_path)
    else : 
        model, aggregator, regressor = load_model(
            config, model, aggregator, regressor, model_path)
    convert_models_to_fp32(model)
    model = weight_mode(model, trainable=False)
    model.eval()
    aggregator = weight_mode(aggregator, trainable=False)
    aggregator.to(device)
    convert_models_to_fp32(aggregator)
    aggregator.eval()
    regressor = weight_mode(regressor, trainable=False)
    regressor.to(device)
    convert_models_to_fp32(regressor)
    regressor.eval()

    print(config)
    pooled_test_loader = init_test_dataloaders(config)

    with torch.no_grad():
        test_prediction, corresponding_mos, corresponding_name = get_predictions(
            config, pooled_test_loader, model, aggregator, regressor, device, tokenizer, image_processor)

    srcc_test_correlation = spearmanr(
        np.array(test_prediction), np.array(corresponding_mos))[0]
    plcc_test_correlation = pearsonr(
        np.array(test_prediction), np.array(corresponding_mos))[0]

    print(f"Performance on {config.test_dataset} is {srcc_test_correlation}")
    print(f"Performance on {config.test_dataset} is {plcc_test_correlation}")

    return srcc_test_correlation, plcc_test_correlation, test_prediction, corresponding_mos, corresponding_name


def configuration_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--embed_dim', type=int, default=4096)
    parser.add_argument('--network_type', type=str, default='b_attn')
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--default_device', type=int, default=7)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--model_description', type=str, default='internlm_vl')
    config = parser.parse_args()
    return config


def get_dataset_info():
    return {

        # 'live_iqa': {
        #     'json_path': '/home/sanjotst/llm_iqa/llm-iqa/labels/live_iqa.json',
        #     'img_dir': '/scratch/sanjotst/datasets/LIVE_IQA'
        # },
        'live_iqa': {
            'json_path': '/scratch/sanjotst/datasets/LIVE_IQA_new/live_iqa_new.json',
            'img_dir': '/scratch/sanjotst/datasets/LIVE_IQA_new'
        },
        'nnid': {
            'json_path': '/scratch/sanjotst/datasets/NNID/nnid.json',
            'img_dir': '/scratch/sanjotst/datasets/NNID'
        },
        'livec': {
            'json_path': '/home/sanjotst/llm_iqa/llm-iqa/labels/livec.json',
            'img_dir': '/scratch/sanjotst/datasets/CLIVE/ChallengeDB_release/Images'
        },
        'kadid': {
            'json_path': '/home/sanjotst/llm_iqa/llm-iqa/labels/kadid.json',
            'img_dir': '/scratch/sanjotst/datasets/kadid10k/images'
        },
        'koniq': {
            'json_path': '/home/sanjotst/llm_iqa/llm-iqa/labels/koniq.json',
            'img_dir': '/scratch/sanjotst/datasets/KonIQ-10k/512x384'
        },
        'pipal': {
            'json_path': '/home/sanjotst/llm_iqa/llm-iqa/labels/pipal.json',
            'img_dir': '/scratch/sanjotst/datasets/PIPAL'
        },
        'spaq': {
            'json_path': '/home/sanjotst/llm_iqa/llm-iqa/labels/spaq.json',
            'img_dir': '/scratch/sanjotst/datasets/SPAQ_512/TestImage'
        }
    }


def test_all_datasets(config):
    dataset_info = get_dataset_info()
    results = {}

    model_name = os.path.basename(
        os.path.dirname(os.path.dirname(config.model_path)))
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{model_name}_{current_time}"

    test_results_dir = os.path.dirname(os.path.dirname(config.model_path))
    results_dir = os.path.join(test_results_dir, folder_name)
    os.makedirs(results_dir, exist_ok=True)

    for dataset_name, dataset_paths in dataset_info.items():
        config.test_dataset = [dataset_name]
        config.img_dir = dataset_paths['img_dir']
        config.input_json_file = dataset_paths['json_path']

        srcc, plcc, predicted_scores, corresponding_mos, corresponding_name = test_model(
            config)
        results[dataset_name] = {'SRCC': srcc, 'PLCC': plcc}
        save_predictions_to_csv(
            predicted_scores, corresponding_mos, corresponding_name, dataset_name, results_dir)

    return results


def save_results(config, results):
    model_name = os.path.basename(
        os.path.dirname(os.path.dirname(config.model_path)))
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{model_name}_{current_time}"

    test_results_dir = os.path.dirname(os.path.dirname(config.model_path))
    results_dir = os.path.join(test_results_dir, folder_name)
    os.makedirs(results_dir, exist_ok=True)

    file_name = f"results_{folder_name}.txt"
    file_path = os.path.join(results_dir, file_name)

    with open(file_path, 'w') as f:
        for dataset, scores in results.items():
            f.write(f"{dataset}:\n")
            f.write(f"  SRCC: {scores['SRCC']}\n")
            f.write(f"  PLCC: {scores['PLCC']}\n\n")

    print(f"Results saved to: {file_path}")


def main():
    config = configuration_params()

    # Load the configuration details from the json
    config_path = os.path.join(os.path.dirname(
        os.path.dirname(config.model_path)), "config_details.json")
    try:
        with open(config_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Config file '{config_path}' not found. Exiting.")
        return

    for key in data:
        if not hasattr(config, key):
            setattr(config, key, data[key])

    results = test_all_datasets(config)
    save_results(config, results)


def main():
    config = configuration_params()

    config_path = os.path.join(os.path.dirname(
        os.path.dirname(config.model_path)), "config_details.json")
    try:
        with open(config_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Config file '{config_path}' not found. Exiting.")
        return

    for key in data:
        if not hasattr(config, key):
            setattr(config, key, data[key])

    results = test_all_datasets(config)
    save_results(config, results)


def run_program():
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

    return run_result


if __name__ == '__main__':
    run_result = run_program()
    print(run_result)
