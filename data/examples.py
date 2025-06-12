import os
import torch
from data.prepare_data import prepare_federated_dataset, visualize_data_distribution, validate_non_iid

def prepare_image_classification_task(
    num_clients: int = 10,
    dataset_name: str = 'cifar10',
    alpha: float = 0.5
):
    """Prepare CIFAR-10/MNIST dataset for CNN-based federated learning"""
    
    print(f"Preparing {dataset_name} dataset for {num_clients} clients...")
    dataset = prepare_federated_dataset(
        dataset_type='image',
        num_clients=num_clients,
        alpha=alpha,
        root='./data/raw',
        dataset_name=dataset_name
    )
    
    # Visualize data distribution
    os.makedirs('./data/plots', exist_ok=True)
    visualize_data_distribution(
        dataset,
        save_path=f'./data/plots/{dataset_name}_distribution.png'
    )
    
    # Validate non-IID property
    is_non_iid = validate_non_iid(dataset)
    print(f"Dataset is sufficiently non-IID: {is_non_iid}")
    
    return dataset

def prepare_text_classification_task(
    num_clients: int = 10,
    alpha: float = 0.5
):
    """
    Prepare text dataset for transformer-based federated learning
    Using a sample sentiment classification task
    """
    # Sample data (replace with your actual data)
    texts = [
        "This is a positive review",
        "I did not like this product",
        # Add more examples...
    ]
    labels = [1, 0]  # Binary classification
    
    print(f"Preparing text classification dataset for {num_clients} clients...")
    dataset = prepare_federated_dataset(
        dataset_type='text',
        num_clients=num_clients,
        alpha=alpha,
        texts=texts,
        labels=labels,
        model_name='bert-base-uncased'
    )
    
    return dataset

def prepare_time_series_task(
    num_clients: int = 10,
    alpha: float = 0.5,
    sequence_length: int = 50
):
    """
    Prepare time series dataset for RNN-based federated learning
    Using synthetic data as an example
    """
    # Generate synthetic time series data (replace with your actual data)
    num_samples = 1000
    num_timesteps = 200
    num_features = 10
    
    sequences = torch.randn(num_samples, num_timesteps, num_features)
    labels = torch.randint(0, 2, (num_samples,))  # Binary classification
    
    print(f"Preparing time series dataset for {num_clients} clients...")
    dataset = prepare_federated_dataset(
        dataset_type='timeseries',
        num_clients=num_clients,
        alpha=alpha,
        sequences=sequences,
        labels=labels,
        sequence_length=sequence_length
    )
    
    return dataset

if __name__ == "__main__":
    # Example usage
    num_clients = 10
    alpha = 0.1  # Low alpha for high non-IID
    
    # Prepare datasets for different architectures
    image_dataset = prepare_image_classification_task(
        num_clients=num_clients,
        dataset_name='cifar10',
        alpha=alpha
    )
    
    text_dataset = prepare_text_classification_task(
        num_clients=num_clients,
        alpha=alpha
    )
    
    time_series_dataset = prepare_time_series_task(
        num_clients=num_clients,
        alpha=alpha,
        sequence_length=50
    )
    
    print("\nDataset Statistics:")
    print("------------------")
    print(f"Image dataset - CIFAR10:")
    print(f"Input shape: {image_dataset.input_shape}")
    print(f"Number of clients: {num_clients}")
    
    print(f"\nText dataset:")
    print(f"Vocabulary size: {text_dataset.vocab_size}")
    print(f"Number of clients: {num_clients}")
    
    print(f"\nTime series dataset:")
    print(f"Input shape: {time_series_dataset.input_shape}")
    print(f"Number of clients: {num_clients}")
