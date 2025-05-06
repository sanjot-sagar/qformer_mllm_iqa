
# No-Reference Image Quality Assessment (NR-IQA) using Multimodal Models

This repository performs inference for image quality assessment on AI-generated images using a pre-trained multimodal model.

## 🔧 Installation

### Step 1: Create and activate a conda environment

```bash
conda create -n inference python=3.10
conda activate inference

### Step 2: Install dependencies

Make sure `requirements.txt` is present in the root directory.

```bash
pip install -r requirements.txt
```

---

## 📁 Dataset Structure

To run inference, you must update the dataset details in the `get_dataset_info` function.

Here is an example configuration for the AGIQA-3k dataset:

```python
def get_dataset_info():
    return {
        'AGIQA-3k': {
            'json_path': '/scratch/sanjotst/datasets/AGIQA-3k/agiqa3k.json',
            'img_dir': '/scratch/sanjotst/datasets/AGIQA-3k/Images'
        }
    }
```

The `.json` file must contain a list of dictionaries with the following format:

```json
[
    {
        "img_path": "/scratch/sanjotst/datasets/AGIQA-3k/Images/AttnGAN_normal_000.jpg",
        "gt_score": 0.965602553
    },
    ...
]
```

---

## ⚙️ Inference Parameters

The script uses command-line arguments to control the inference process. These are defined in the `configuration_params` function:

| Parameter             | Description                                              |
| --------------------- | -------------------------------------------------------- |
| `--mode`              | Mode of operation (`eval` for evaluation)                |
| `--model_path`        | Path to the `.tar` model file (e.g., `iter_5673.tar`)    |
| `--embed_dim`         | Embedding dimension used in the model                    |
| `--network_type`      | Network architecture type (e.g., `b_attn`)               |
| `--num_heads`         | Number of attention heads (for models using attention)   |
| `--default_device`    | GPU device ID to run inference on                        |
| `--num_gpus`          | Number of GPUs to use (typically 1)                      |
| `--model_description` | A string describing the base model (e.g., `internlm_vl`) |
| `--model_type`        | Custom identifier for model variant (e.g., `no_llm`)     |

---

## 📦 Model Files

The model weights (e.g., `iter_5673.tar`) are available on Google Drive. You must download and place the **entire folder** (not just the `.tar` file) in your working directory. This is because the inference script loads the training configuration from `config_details.json`, which resides in the same folder hierarchy as the model file.
Get the files : https://drive.google.com/drive/folders/1rf8AmVWnpYgKbe2zcJx6AkMxsZ8WPEbk?usp=sharing
The model weights file iter_5673.tar is inside train folder. You need to download the Run0004 folder. 
```python
config_path = os.path.join(os.path.dirname(
    os.path.dirname(config.model_path)), "config_details.json")
```

Make sure this file is present alongside your model directory structure.

---

## 🚀 Running Inference

Once the environment is set up and files are in place, run the testing.py script with the required parameters. To replicate results with inference, you need to run with the exact parameters as given, while changing the GPU device ID and model_path. 

---

```

```
