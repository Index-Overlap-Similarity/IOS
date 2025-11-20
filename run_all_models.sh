#!/bin/bash

# Script to run IOS ML Benchmark for all 4 models
# Usage: ./run_all_models.sh [quick|full]

MODE=${1:-full}  # Default to 'full', can be 'quick'

echo "=========================================="
echo "IOS ML Benchmark - Running All 4 Models"
echo "Mode: $MODE"
echo "=========================================="
echo ""

# Function to run a model
run_model() {
    local model=$1
    local dataset=$2
    local log_file=$3
    local extra_args=$4
    
    echo "----------------------------------------"
    echo "Starting: $model on $dataset"
    echo "Log file: $log_file"
    echo "----------------------------------------"
    
    python main.py \
        --model $model \
        --dataset $dataset \
        --log_file $log_file \
        $extra_args
    
    if [ $? -eq 0 ]; then
        echo "✓ $model completed successfully!"
    else
        echo "✗ $model failed!"
        return 1
    fi
    echo ""
}

# Quick mode settings
if [ "$MODE" == "quick" ]; then
    echo "Running in QUICK mode (reduced clients and batches)..."
    echo ""
    
    # Model 1: CNN
    run_model "cnn" "fmnist" "cnn_fmnist_quick.log" \
        "--num_clients 1 --max_batches_per_client 10 --bandwidth_mbps 200"
    
    # Model 2: ResNet18
    run_model "resnet18" "cifar10" "resnet18_cifar10_quick.log" \
        "--num_clients 1 --max_batches_per_client 5 --bandwidth_mbps 200"
    
    # Model 3: ResNet50
    run_model "resnet50" "cifar10" "resnet50_cifar10_quick.log" \
        "--num_clients 1 --max_batches_per_client 3 --bandwidth_mbps 200 --batch_size 32"
    
    # Model 4: BERT Base
    run_model "bert_base" "20newsgroups" "bert_base_20newsgroups_quick.log" \
        "--num_clients 1 --max_batches_per_client 2 --bandwidth_mbps 200 --batch_size 8"
    
else
    echo "Running in FULL mode (default settings)..."
    echo ""
    
    # Model 1: CNN
    run_model "cnn" "fmnist" "cnn_fmnist.log" ""
    
    # Model 2: ResNet18
    run_model "resnet18" "cifar10" "resnet18_cifar10.log" ""
    
    # Model 3: ResNet50
    run_model "resnet50" "cifar10" "resnet50_cifar10.log" "--batch_size 32"
    
    # Model 4: BERT Base
    run_model "bert_base" "20newsgroups" "bert_base_20newsgroups.log" "--batch_size 16"
fi

echo "=========================================="
echo "All models completed!"
echo "=========================================="
echo ""
echo "Results saved in: ios_outputs/"
echo "Log files: *.log"
echo ""
echo "To view results:"
echo "  cat ios_outputs/results.json"
echo ""

