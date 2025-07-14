# GenomicTouchstone

A comprehensive deep learning framework for genomic sequence analysis and benchmarking. This project provides a unified platform for fine-tuning and evaluating various pre-trained genomic foundation models on downstream tasks.

## 🧬 Overview

GenomicTouchstone is designed to benchmark and fine-tune state-of-the-art genomic foundation models on diverse genomic tasks. It supports multiple model architectures and provides standardized evaluation protocols for fair comparison across different approaches.

## 🚀 Quick Start

### Environment Setup

1. **Create conda environment:**
```bash
conda env create -f Genomictouchstone.yml
conda activate Genomictouchstone
```

2. **Install additional dependencies:**
```bash
# For flash attention
pip install flash-attn==2.5.6
pip install causal-conv1d==1.2.0.post2
pip install mamba-ssm==1.2.0.post1
```

### Data Preparation

Download the GenomicTouchstone datasets from [Hugging Face](https://huggingface.co/datasets/Wangyh/GenomicTouchstone_raw):

```bash
git clone https://huggingface.co/datasets/Wangyh/GenomicTouchstone_raw ./data/GenomicTouchstone_raw
```

### Basic Usage

#### 1. Single Model Training
```bash
python -m train experiment=genomictouchstone/classification \
    dataset.dataset_name="core_promoter_annotation" \
    model=dnabert \
    train.pretrained_model_path="/path/to/pretrained/model" \
    trainer.devices=1
```

#### 2. Batch Training with Scripts
```bash
# Navigate to classification scripts
cd finetune/Genomictouchstone/classification/

# Run comprehensive benchmarking
bash classification.sh
```

### 📝 Customizing Training Scripts

The `classification.sh` script is highly configurable and supports comprehensive benchmarking across multiple models and tasks. 

Here's how to customize it:

#### 1. **Configure Data and Model Paths**

Edit the hostname-specific paths at the beginning of the script:

```bash
model_path="/your/path/to/models"
data_path="/your/path/to/data"
```

We recommend to download the pretrained models from the official model zoo and put them in the `model_path` directory.

For example, if you want to run the nucleotide-transformer-2.5b-multi-species model

```bash
cd /your/path/to/models
# Make sure git-lfs is installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species
```

#### 2. **Select Tasks to Run**

Modify the `TASKs` variable to choose which benchmarks to run:

```bash
TASKs='core_promoter_annotation'
```

#### 3. **Configure Model Execution**

At the bottom of the script, uncomment the models you want to run. Currently active:

```bash
# run_model is the function to run the model, you can modify the model name and the gpu id to run the model
run_model "$CONFIG_DNABERT3" 0
```

#### 4. **Add Custom Models**

To add a new model configuration:

```bash
# Define model configuration
CONFIG_YOUR_MODEL="your_model"
PRETRAINED_PATHS[$CONFIG_YOUR_MODEL]="${model_path}/your-model-path"
DISPLAY_NAMES[$CONFIG_YOUR_MODEL]="your-model-display-name"
MODELS[$CONFIG_YOUR_MODEL]="your_model_type"
TOKENIZERS[$CONFIG_YOUR_MODEL]="your_tokenizer"
MODEL_NAMES[$CONFIG_YOUR_MODEL]="dna_embedding_your_model"
D_MODELS[$CONFIG_YOUR_MODEL]=768
LRs[$CONFIG_YOUR_MODEL]=3e-5
BATCH_SIZES[$CONFIG_YOUR_MODEL]=32
MAX_LENGTHS[$CONFIG_YOUR_MODEL]=512
RETURN_MASKs[$CONFIG_YOUR_MODEL]=true

# Add to execution section
run_model "$CONFIG_YOUR_MODEL" 0
```



## 📄 License

This project is licensed under the CC-BY-NC-ND 4.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This repository is adapted from the [HyenaDNA repo](https://github.com/HazyResearch/hyena-dna) and leverages much of the training, data loading, and logging infrastructure defined there. We also thank the [Caduceus repository](https://github.com/kuleshov-group/caduceus) for additional model implementations and insights.


---

For more detailed documentation and examples, please refer to the configuration files and scripts in the repository.
