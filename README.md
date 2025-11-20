# IOS
IOS source code

This guide provides step-by-step instructions to run the IOS ML Benchmark with all four models: CNN, ResNet18, ResNet50, and BERT Base.

## Prerequisites

### 1. Install Required Dependencies

```bash
# Use UV with
uv sync # Or

# Core dependencies (should already be installed)
pip install torch torchvision numpy

# For BERT Base only
pip install transformers
```

### 2. Download Datasets

The datasets will be automatically downloaded on first run, but you can verify:
- FashionMNIST: `./data/FashionMNIST/`
- CIFAR-10: `./data/cifar-10-batches-py/`
- CIFAR-100: `./data/cifar-100-python/`

---

## Model 1: CNN (SimpleCNN)

**Best for:** FashionMNIST dataset  
**Use case:** Quick experiments, smaller models

### Basic Run
```bash
python main.py \
    --model cnn \
    --dataset fmnist \
    --log_file cnn_fmnist.log
```

### Full Configuration
```bash
python main.py \
    --model cnn \
    --dataset fmnist \
    --num_clients 15 \
    --batch_size 64 \
    --partition dir \
    --dir_alpha 0.3 \
    --local_epochs 1 \
    --lr 0.001 \
    --topk_frac 0.12 \
    --usecases all \
    --log_file cnn_fmnist_full.log
```

### Quick Test (Fast)
```bash
python main.py \
    --model cnn \
    --dataset fmnist \
    --num_clients 5 \
    --max_batches_per_client 10 \
    --local_epochs 1 \
    --log_file cnn_fmnist_quick.log
```

**Expected Runtime:** ~2-5 minutes (depending on hardware)  
**Output Files:**
- `ios_outputs/results.json`
- `ios_outputs/S_ios.npy`
- `ios_outputs/S_cos.npy`
- `ios_outputs/S_oracle.npy`
- `cnn_fmnist.log` (if specified)

---

## Model 2: ResNet18

**Best for:** CIFAR-10 dataset  
**Use case:** Standard baseline, good balance of performance and speed

### Basic Run
```bash
python main.py \
    --model resnet18 \
    --dataset cifar10 \
    --log_file resnet18_cifar10.log
```

### Full Configuration
```bash
python main.py \
    --model resnet18 \
    --dataset cifar10 \
    --num_clients 15 \
    --batch_size 64 \
    --partition dir \
    --dir_alpha 0.3 \
    --local_epochs 3 \
    --lr 0.001 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --topk_frac 0.12 \
    --pretrain_global_epochs 0 \
    --usecases all \
    --k_neighbors 5 \
    --log_file resnet18_cifar10_full.log
```

### With Global Pretraining
```bash
python main.py \
    --model resnet18 \
    --dataset cifar10 \
    --pretrain_global_epochs 5 \
    --local_epochs 3 \
    --log_file resnet18_cifar10_pretrained.log
```

### Quick Test (Fast)
```bash
python main.py \
    --model resnet18 \
    --dataset cifar10 \
    --num_clients 5 \
    --max_batches_per_client 5 \
    --local_epochs 1 \
    --log_file resnet18_cifar10_quick.log
```

**Expected Runtime:** ~10-30 minutes (depending on hardware and epochs)  
**Output Files:**
- `ios_outputs/results.json`
- `ios_outputs/S_ios.npy`
- `ios_outputs/S_cos.npy`
- `ios_outputs/S_oracle.npy`
- `resnet18_cifar10.log` (if specified)

---

## Model 3: ResNet50

**Best for:** CIFAR-10 or CIFAR-100 dataset  
**Use case:** Larger model, more parameters, better representation

### Basic Run
```bash
python main.py \
    --model resnet50 \
    --dataset cifar10 \
    --log_file resnet50_cifar10.log
```

### Full Configuration
```bash
python main.py \
    --model resnet50 \
    --dataset cifar10 \
    --num_clients 15 \
    --batch_size 32 \
    --partition dir \
    --dir_alpha 0.3 \
    --local_epochs 3 \
    --lr 0.001 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --topk_frac 0.12 \
    --pretrain_global_epochs 0 \
    --usecases all \
    --k_neighbors 5 \
    --log_file resnet50_cifar10_full.log
```

### With CIFAR-100 (More Classes)
```bash
python main.py \
    --model resnet50 \
    --dataset cifar100 \
    --num_clients 15 \
    --local_epochs 3 \
    --log_file resnet50_cifar100.log
```

### Quick Test (Fast)
```bash
python main.py \
    --model resnet50 \
    --dataset cifar10 \
    --num_clients 5 \
    --max_batches_per_client 3 \
    --local_epochs 1 \
    --batch_size 32 \
    --log_file resnet50_cifar10_quick.log
```

**Note:** ResNet50 is larger, so use smaller batch sizes (32) if you run into memory issues.

**Expected Runtime:** ~30-60 minutes (depending on hardware)  
**Output Files:**
- `ios_outputs/results.json`
- `ios_outputs/S_ios.npy`
- `ios_outputs/S_cos.npy`
- `ios_outputs/S_oracle.npy`
- `resnet50_cifar10.log` (if specified)

---

## Model 4: BERT Base

**Best for:** Text classification tasks (Note: Current implementation uses CIFAR-10, but BERT is designed for text)  
**Use case:** Transformer-based model, requires transformers library

### Prerequisites Check
```bash
# Verify transformers is installed
python -c "import transformers; print(transformers.__version__)"
```

### Basic Run
```bash
python main.py \
    --model bert_base \
    --dataset cifar10 \
    --log_file bert_base_cifar10.log
```

### Full Configuration
```bash
python main.py \
    --model bert_base \
    --dataset cifar10 \
    --num_clients 15 \
    --batch_size 16 \
    --partition dir \
    --dir_alpha 0.3 \
    --local_epochs 2 \
    --lr 2e-5 \
    --momentum 0.9 \
    --weight_decay 0.01 \
    --topk_frac 0.12 \
    --pretrain_global_epochs 0 \
    --usecases all \
    --k_neighbors 5 \
    --log_file bert_base_cifar10_full.log
```

### Quick Test (Fast)
```bash
python main.py \
    --model bert_base \
    --dataset cifar10 \
    --num_clients 5 \
    --max_batches_per_client 2 \
    --local_epochs 1 \
    --batch_size 8 \
    --log_file bert_base_cifar10_quick.log
```

**Note:** 
- BERT Base is very large (~110M parameters)
- Use smaller batch sizes (8-16) to avoid OOM errors
- BERT typically uses lower learning rates (2e-5 to 5e-5)
- This model may take significantly longer to run

**Expected Runtime:** ~1-3 hours (depending on hardware)  
**Output Files:**
- `ios_outputs/results.json`
- `ios_outputs/S_ios.npy`
- `ios_outputs/S_cos.npy`
- `ios_outputs/S_oracle.npy`
- `bert_base_cifar10.log` (if specified)

---

## Running All Models in Sequence

### Script to Run All Models

Create a file `run_all_models.sh`:

```bash
#!/bin/bash

echo "=========================================="
echo "Running IOS Benchmark for All 4 Models"
echo "=========================================="

# Model 1: CNN
echo "Starting CNN (FashionMNIST)..."
python main.py --model cnn --dataset fmnist --log_file cnn_fmnist.log
echo "CNN completed!"

# Model 2: ResNet18
echo "Starting ResNet18 (CIFAR-10)..."
python main.py --model resnet18 --dataset cifar10 --log_file resnet18_cifar10.log
echo "ResNet18 completed!"

# Model 3: ResNet50
echo "Starting ResNet50 (CIFAR-10)..."
python main.py --model resnet50 --dataset cifar10 --log_file resnet50_cifar10.log
echo "ResNet50 completed!"

# Model 4: BERT Base
echo "Starting BERT Base (CIFAR-10)..."
python main.py --model bert_base --dataset cifar10 --log_file bert_base_cifar10.log
echo "BERT Base completed!"

echo "=========================================="
echo "All models completed!"
echo "=========================================="
```

Make it executable and run:
```bash
chmod +x run_all_models.sh
./run_all_models.sh
```

---

## Quick Comparison Run (All Models, Fast Settings)

For quick comparison across all models:

```bash
# CNN - Quick
python main.py --model cnn --dataset fmnist --num_clients 5 --max_batches_per_client 10 --log_file cnn_quick.log

# ResNet18 - Quick
python main.py --model resnet18 --dataset cifar10 --num_clients 5 --max_batches_per_client 5 --log_file resnet18_quick.log

# ResNet50 - Quick
python main.py --model resnet50 --dataset cifar10 --num_clients 5 --max_batches_per_client 3 --batch_size 32 --log_file resnet50_quick.log

# BERT Base - Quick
python main.py --model bert_base --dataset cifar10 --num_clients 5 --max_batches_per_client 2 --batch_size 8 --log_file bert_quick.log
```

---

## Understanding the Output

### Console Output
- Real-time logging of each step
- Timing information for each operation
- Final results summary

### Log File (if `--log_file` specified)
- Complete step-by-step log
- All timing information
- Detailed progress for each client

### Results JSON (`ios_outputs/results.json`)
Contains metrics like:
- `spearman_ios_vs_oracle`: Correlation between IOS and oracle similarities
- `spearman_cos_vs_oracle`: Correlation between cosine and oracle similarities
- `topk_recall_ios@k`: Top-k recall for IOS
- `neighbor_js_ios@k`: Neighbor selection JS divergence
- `drift_auroc`: Drift detection AUROC
- And more...

### Similarity Matrices
- `S_ios.npy`: IOS similarity matrix (Jaccard on top-k sets)
- `S_cos.npy`: Cosine similarity matrix (on full vectors)
- `S_oracle.npy`: Oracle similarity matrix (1 - JS divergence on label distributions)

---

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `--batch_size` (try 16 or 8)
- Reduce `--num_clients`
- Use `--max_batches_per_client` to limit batches
- For BERT: use batch size 8 or 4

### Slow Performance
- Use `--max_batches_per_client` for quick tests
- Reduce `--local_epochs`
- Reduce `--num_clients`
- Use smaller models (CNN or ResNet18) for faster iteration

### BERT Base Issues
- Ensure `transformers` library is installed: `pip install transformers`
- Use smaller batch sizes (8-16)
- May need to adjust learning rate (try 2e-5)

### GPU Not Being Used
- Check line 1056 in `main.py` - change `cuda:3` to your GPU ID (e.g., `cuda:0`)
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

---

## Tips

1. **Start Small**: Use quick test configurations first to verify everything works
2. **Monitor Logs**: Watch the log files to track progress and identify bottlenecks
3. **Compare Models**: Run all models with same settings to compare performance
4. **Save Results**: Each run creates timestamped outputs in `ios_outputs/`
5. **Use Log Files**: Always use `--log_file` for production runs to keep detailed records

---

## Example: Complete Workflow

```bash
# 1. Quick test all models
python main.py --model cnn --dataset fmnist --num_clients 3 --max_batches_per_client 5 --log_file test_cnn.log
python main.py --model resnet18 --dataset cifar10 --num_clients 3 --max_batches_per_client 3 --log_file test_resnet18.log
python main.py --model resnet50 --dataset cifar10 --num_clients 3 --max_batches_per_client 2 --batch_size 16 --log_file test_resnet50.log
python main.py --model bert_base --dataset cifar10 --num_clients 3 --max_batches_per_client 1 --batch_size 4 --log_file test_bert.log

# 2. Full runs (after quick tests pass)
python main.py --model cnn --dataset fmnist --log_file cnn_full.log
python main.py --model resnet18 --dataset cifar10 --log_file resnet18_full.log
python main.py --model resnet50 --dataset cifar10 --log_file resnet50_full.log
python main.py --model bert_base --dataset cifar10 --batch_size 16 --log_file bert_full.log

# 3. Compare results
cat ios_outputs/results.json
```

---

