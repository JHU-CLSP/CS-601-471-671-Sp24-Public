import gensim.downloader
from easydict import EasyDict
from mlp import (
    run_mlp,
    load_data_mlp,
    visualize_configs,
    visualize_epochs,
    create_tensor_dataset,
)
from typing import List, Tuple, Dict, Union

EMBEDDING_TYPE = "glove-twitter-50"

def explore_mlp_structures(
    dev_d: Dict[str, List[Union[str, int]]],
    train_d: Dict[str, List[Union[str, int]]],
    test_d: Dict[str, List[Union[str, int]]],
    embeddings: gensim.models.keyedvectors.Word2VecKeyedVectors,
):
    """
    Explore the effect of different MLP structures on model performance.
    
    Parameters:
        dev_d: Development dataset.
        train_d: Training dataset.
        test_d: Test dataset.
        embeddings: Pre-trained word embeddings.
    """
    all_emb_epoch_dev_accs, all_emb_epoch_dev_losses = [], []

    # Create datasets from raw data
    train_dataset = create_tensor_dataset(train_d, embeddings)
    dev_dataset = create_tensor_dataset(dev_d, embeddings)
    test_dataset = create_tensor_dataset(test_d, embeddings)

    # Different configurations of hidden layers
    hidden_dims = [[], [512], [512, 512], [512, 512, 512]]
    hidden_dims_names = ["None", "512", "512 -> 512", "512 -> 512 -> 512"]  # for visualization
    learning_rates = [0.025, 0.02, 0.01, 0.001]  # learning rates for each configuration

    # Fixed hyperparameters
    batch_size = 64
    num_epochs = 20
    activation = "Sigmoid"
    num_classes = 2

    for hd, hdn, lr in zip(hidden_dims, hidden_dims_names, learning_rates):
        # Configure training settings
        train_config = EasyDict({
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "hidden_dims": hd,
            "save_path": f"model_hidden_{hdn}.pth",
            "embeddings": EMBEDDING_TYPE,
            "num_classes": num_classes,
            "activation": activation,
        })

        # Run training and collect results
        epoch_train_losses, _, epoch_dev_loss, epoch_dev_accs, _, _ = run_mlp(
            train_config, embeddings, dev_dataset, train_dataset, test_dataset
        )
        all_emb_epoch_dev_accs.append(epoch_dev_accs)
        all_emb_epoch_dev_losses.append(epoch_dev_loss)
        
        # Visualize loss per epoch for the current configuration
        visualize_epochs(
            epoch_train_losses, epoch_dev_loss, f"mlp_structure_{hdn}_loss.png"
        )

    # Visualize overall accuracy and loss across all configurations
    visualize_configs(
        all_emb_epoch_dev_accs, hidden_dims_names, "Accuracy", "./all_mlp_structures_acc.png",
    )
    visualize_configs(
        all_emb_epoch_dev_losses, hidden_dims_names, "Loss", "./all_mlp_structures_loss.png",
    )

def explore_mlp_activations(
    dev_d: Dict[str, List[Union[str, int]]],
    train_d: Dict[str, List[Union[str, int]]],
    test_d: Dict[str, List[Union[str, int]]],
    embeddings: gensim.models.keyedvectors.Word2VecKeyedVectors,
):
    """
    Explore the effect of different activation functions on MLP performance.
    
    Parameters:
        dev_d: Development dataset.
        train_d: Training dataset.
        test_d: Test dataset.
        embeddings: Pre-trained word embeddings.
    """
    all_emb_epoch_dev_accs, all_emb_epoch_dev_losses = [], []

    # Create datasets from raw data
    train_dataset = create_tensor_dataset(train_d, embeddings)
    dev_dataset = create_tensor_dataset(dev_d, embeddings)
    test_dataset = create_tensor_dataset(test_d, embeddings)

    # Fixed hyperparameters
    batch_size = 128
    num_epochs = 20
    lr = 0.02
    num_classes = 2
    hidden_dims = [512]

    # Different activation functions to explore
    activation_functions = ["Sigmoid", "Tanh", "ReLU", "GeLU"]
    activation_names = activation_functions  # for visualization

    for activation_func, activation_name in zip(activation_functions, activation_names):
        # Configure training settings
        train_config = EasyDict({
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "hidden_dims": hidden_dims,
            "save_path": f"model_activation_{activation_name}.pth",
            "embeddings": EMBEDDING_TYPE,
           

 "num_classes": num_classes,
            "activation": activation_func,
        })

        # Run training and collect results
        epoch_train_losses, _, epoch_dev_loss, epoch_dev_accs, _, _ = run_mlp(
            train_config, embeddings, dev_dataset, train_dataset, test_dataset
        )
        all_emb_epoch_dev_accs.append(epoch_dev_accs)
        all_emb_epoch_dev_losses.append(epoch_dev_loss)
        
        # Visualize loss per epoch for the current activation function
        visualize_epochs(
            epoch_train_losses, epoch_dev_loss, f"mlp_activation_{activation_name}_loss.png",
        )

    # Visualize overall accuracy and loss across all activation functions
    visualize_configs(
        all_emb_epoch_dev_accs, activation_names, "Accuracy", "./all_mlp_activations_acc.png",
    )
    visualize_configs(
        all_emb_epoch_dev_losses, activation_names, "Loss", "./all_mlp_activations_loss.png",
    )

def explore_mlp_learning_rates(
    dev_d: Dict[str, List[Union[str, int]]],
    train_d: Dict[str, List[Union[str, int]]],
    test_d: Dict[str, List[Union[str, int]]],
    embeddings: gensim.models.keyedvectors.Word2VecKeyedVectors,
):
    """
    Explore the effect of different learning rates on MLP performance.
    
    Parameters:
        dev_d: Development dataset.
        train_d: Training dataset.
        test_d: Test dataset.
        embeddings: Pre-trained word embeddings.
    """
    all_emb_epoch_dev_accs, all_emb_epoch_dev_losses = [], []

    # Create datasets from raw data
    train_dataset = create_tensor_dataset(train_d, embeddings)
    dev_dataset = create_tensor_dataset(dev_d, embeddings)
    test_dataset = create_tensor_dataset(test_d, embeddings)

    # Fixed hyperparameters
    batch_size = 64
    num_epochs = 20
    num_classes = 2
    hidden_dims = [512]
    activation = "Sigmoid"

    # Different learning rates to explore
    lrs = [0.02, 0.05, 0.01, 0.03, 0.1]
    lrs_names = [str(lr) for lr in lrs]  # for visualization

    for lr, lr_name in zip(lrs, lrs_names):
        # Configure training settings
        train_config = EasyDict({
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "hidden_dims": hidden_dims,
            "save_path": f"model_lr_{lr_name}.pth",
            "embeddings": EMBEDDING_TYPE,
            "num_classes": num_classes,
            "activation": activation,
        })

        # Run training and collect results
        epoch_train_losses, _, epoch_dev_loss, epoch_dev_accs, _, _ = run_mlp(
            train_config, embeddings, dev_dataset, train_dataset, test_dataset
        )
        all_emb_epoch_dev_accs.append(epoch_dev_accs)
        all_emb_epoch_dev_losses.append(epoch_dev_loss)
        
        # Visualize loss per epoch for the current learning rate
        visualize_epochs(
            epoch_train_losses, epoch_dev_loss, f"mlp_lr_{lr_name}_loss.png"
        )

    # Visualize overall accuracy and loss across all learning rates
    visualize_configs(
        all_emb_epoch_dev_accs, lrs_names, "Accuracy", "./all_mlp_learning_rates_acc.png",
    )
    visualize_configs(
        all_emb_epoch_dev_losses, lrs_names, "Loss", "./all_mlp_learning_rates_loss.png",
    )

if __name__ == "__main__":
    # Load raw data and pre-trained embeddings, then explore different configurations
    dev_data, train_data, test_data = load_data_mlp()
    print(f"{'-' * 10} Load Pre-trained Embeddings: {EMBEDDING_TYPE} {'-' * 10}")
    pretrained_embeddings = gensim.downloader.load(EMBEDDING_TYPE)
    
    # Uncomment the following lines to run exploration functions
    explore_mlp_structures(dev_data, train_data, test_data, pretrained_embeddings)
    explore_mlp_activations(dev_data, train_data, test_data, pretrained_embeddings)
    explore_mlp_learning_rates(dev_data, train_data, test_data, pretrained_embeddings)