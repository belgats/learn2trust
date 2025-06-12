#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data.examples import (
    prepare_image_classification_task,
    prepare_text_classification_task,
    prepare_time_series_task
)

def main():
    # Configuration
    num_clients = 10
    alpha = 0.1  # Higher non-IID setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Image Classification with CNN
    print("Setting up Image Classification task...")
    image_dataset = prepare_image_classification_task(
        num_clients=num_clients,
        dataset_name='cifar10',
        alpha=alpha
    )
    
    # Example: Load one client's data
    client_0_data, client_0_labels = image_dataset.get_client_data(0)
    print(f"Client 0 data shape: {client_0_data.shape}")
    print(f"Client 0 labels shape: {client_0_labels.shape}")
    
    # 2. Text Classification with Transformer
    print("\nSetting up Text Classification task...")
    text_dataset = prepare_text_classification_task(
        num_clients=num_clients,
        alpha=alpha
    )
    
    # Example: Load one client's data
    client_0_text_data = text_dataset.get_client_data(0)
    print(f"Client 0 input_ids shape: {client_0_text_data['input_ids'].shape}")
    print(f"Client 0 attention_mask shape: {client_0_text_data['attention_mask'].shape}")
    
    # 3. Time Series with RNN
    print("\nSetting up Time Series task...")
    time_series_dataset = prepare_time_series_task(
        num_clients=num_clients,
        alpha=alpha,
        sequence_length=50
    )
    
    # Example: Load one client's data
    client_0_seq, client_0_seq_labels = time_series_dataset.get_client_data(0)
    print(f"Client 0 sequence shape: {client_0_seq.shape}")
    print(f"Client 0 sequence labels shape: {client_0_seq_labels.shape}")
    
if __name__ == "__main__":
    main()
