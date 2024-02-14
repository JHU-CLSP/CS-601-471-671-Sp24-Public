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
    Explore different MLP structures by varying the number of hidden layers and neurons in each layer.

    Args:
        dev_d (Dict[str, List[Union[str, int]]]): Development dataset.
        train_d (Dict[str, List[Union[str, int]]]): Training dataset.
        test_d (Dict[str, List[Union[str, int]]]): Test dataset.
        embeddings (gensim.models.keyedvectors.Word2VecKeyedVectors): Pre-trained word embeddings.

    Returns:
        None
    """
    all_emb_epoch_dev_accs, all_emb_epoch_dev_losses = [], []

    print(f"{'-' * 10} Create Datasets {'-' * 10}")
    train_dataset = create_tensor_dataset(train_d, embeddings)
    dev_dataset = create_tensor_dataset(dev_d, embeddings)
    test_dataset = create_tensor_dataset(test_d, embeddings)

    hidden_dims = [[], [512], [512, 512], [512, 512, 512]]
    hidden_dims_names = [
        "None",
        "512",
        "512 -> 512",
        "512 -> 512 -> 512",
    ]  # for visualization
    learning_rates = [
        0.025,
        0.02,
        0.01,
        0.001,
    ]  # learning rates for each hidden layer configuration

    # fixed hyperparameters
    batch_size = 64
    num_epochs = 20
    embedding_type = EMBEDDING_TYPE
    activation = "Sigmoid"
    num_classes = 2

    for hd, hdn, lr in zip(hidden_dims, hidden_dims_names, learning_rates):
        train_config = EasyDict(
            {
                "batch_size": batch_size,  # we use batching for training
                "lr": lr,
                "num_epochs": num_epochs,  # the total number of times all the training data is iterated over
                "hidden_dims": hd,  # the number of neurons in each hidden layer
                "save_path": f"model_hidden_{hdn}.pth",  # path where to save the model
                "embeddings": embedding_type,
                "num_classes": num_classes,
                "activation": activation,  # non-linear activation function
            }
        )

        epoch_train_losses, _, epoch_dev_loss, epoch_dev_accs, _, _ = run_mlp(
            train_config, embeddings, dev_dataset, train_dataset, test_dataset
        )
        all_emb_epoch_dev_accs.append(epoch_dev_accs)
        all_emb_epoch_dev_losses.append(epoch_dev_loss)
        visualize_epochs(
            epoch_train_losses, epoch_dev_loss, f"mlp_structure_{hdn}_loss.png"
        )

    visualize_configs(
        all_emb_epoch_dev_accs,
        hidden_dims_names,
        "Accuracy",
        "./all_mlp_structures_acc.png",
    )
    visualize_configs(
        all_emb_epoch_dev_losses,
        hidden_dims_names,
        "Loss",
        "./all_mlp_structures_loss.png",
    )


def explore_mlp_activations(
    dev_d: Dict[str, List[Union[str, int]]],
    train_d: Dict[str, List[Union[str, int]]],
    test_d: Dict[str, List[Union[str, int]]],
    embeddings: gensim.models.keyedvectors.Word2VecKeyedVectors,
):
    """
    Explore different activation functions in MLP.

    Args:
        dev_d (Dict[str, List[Union[str, int]]]): Development dataset.
        train_d (Dict[str, List[Union[str, int]]]): Training dataset.
        test_d (Dict[str, List[Union[str, int]]]): Test dataset.
        embeddings (gensim.models.keyedvectors.Word2VecKeyedVectors): Pre-trained word embeddings.

    Returns:
        None
    """
    all_emb_epoch_dev_accs, all_emb_epoch_dev_losses = [], []

    print(f"{'-' * 10} Create Datasets {'-' * 10}")
    train_dataset = create_tensor_dataset(train_d, embeddings)
    dev_dataset = create_tensor_dataset(dev_d, embeddings)
    test_dataset = create_tensor_dataset(test_d, embeddings)

    # fixed hyperparameters
    batch_size = 128
    num_epochs = 20
    lr = 0.02
    embedding_type = EMBEDDING_TYPE
    num_classes = 2
    hidden_dims = [512]

    activation_functions = ["Sigmoid", "Tanh", "ReLU", "GeLU"]
    activation_names = activation_functions  # for visualization

    for activation_func, activation_name in zip(activation_functions, activation_names):
        train_config = EasyDict(
            {
                "batch_size": batch_size,
                "lr": lr,
                "num_epochs": num_epochs,
                "hidden_dims": hidden_dims,
                "save_path": f"model_activation_{activation_name}.pth",
                "embeddings": embedding_type,
                "num_classes": num_classes,
                "activation": activation_func,
            }
        )

        epoch_train_losses, _, epoch_dev_loss, epoch_dev_accs, _, _ = run_mlp(
            train_config, embeddings, dev_dataset, train_dataset, test_dataset
        )
        all_emb_epoch_dev_accs.append(epoch_dev_accs)
        all_emb_epoch_dev_losses.append(epoch_dev_loss)
        visualize_epochs(
            epoch_train_losses,
            epoch_dev_loss,
            f"mlp_activation_{activation_name}_loss.png",
        )

    visualize_configs(
        all_emb_epoch_dev_accs,
        activation_names,
        "Accuracy",
        "./all_mlp_activations_acc.png",
    )
    visualize_configs(
        all_emb_epoch_dev_losses,
        activation_names,
        "Loss",
        "./all_mlp_activations_loss.png",
    )


def explore_mlp_learning_rates(
    dev_d: Dict[str, List[Union[str, int]]],
    train_d: Dict[str, List[Union[str, int]]],
    test_d: Dict[str, List[Union[str, int]]],
    embeddings: gensim.models.keyedvectors.Word2VecKeyedVectors,
):
    """
    Explore different learning rates in MLP.

    Args:
        dev_d (Dict[str, List[Union[str, int]]]): Development dataset.
        train_d (Dict[str, List[Union[str, int]]]): Training dataset.
        test_d (Dict[str, List[Union[str, int]]]): Test dataset.
        embeddings (gensim.models.keyedvectors.Word2VecKeyedVectors): Pre-trained word embeddings.

    Returns:
        None
    """
    all_emb_epoch_dev_accs, all_emb_epoch_dev_losses = [], []

    print(f"{'-' * 10} Create Datasets {'-' * 10}")
    train_dataset = create_tensor_dataset(train_d, embeddings)
    dev_dataset = create_tensor_dataset(dev_d, embeddings)
    test_dataset = create_tensor_dataset(test_d, embeddings)

    # fixed hyperparameters
    batch_size = 64
    num_epochs = 20
    embedding_type = EMBEDDING_TYPE
    num_classes = 2
    hidden_dims = [512]
    activation = "Sigmoid"

    lrs = [0.02, 0.05, 0.01, 0.03, 0.1]

    lrs_names = [str(lr) for lr in lrs]  # for visualization

    all_emb_epoch_dev_accs, all_emb_epoch_dev_losses = [], []

    print(f"{'-' * 10} Create Datasets {'-' * 10}")
    train_dataset = create_tensor_dataset(train_d, embeddings)
    dev_dataset = create_tensor_dataset(dev_d, embeddings)
    test_dataset = create_tensor_dataset(test_d, embeddings)

    # fixed hyperparameters
    batch_size = 64
    num_epochs = 20
    embedding_type = EMBEDDING_TYPE
    num_classes = 2
    hidden_dims = [512]
    activation = "Sigmoid"

    for lr, lr_name in zip(lrs, lrs_names):
        train_config = EasyDict(
            {
                "batch_size": batch_size,
                "lr": lr,
                "num_epochs": num_epochs,
                "hidden_dims": hidden_dims,
                "save_path": f"model_lr_{lr_name}.pth",
                "embeddings": embedding_type,
                "num_classes": num_classes,
                "activation": activation,
            }
        )

        epoch_train_losses, _, epoch_dev_loss, epoch_dev_accs, _, _ = run_mlp(
            train_config, embeddings, dev_dataset, train_dataset, test_dataset
        )
        all_emb_epoch_dev_accs.append(epoch_dev_accs)
        all_emb_epoch_dev_losses.append(epoch_dev_loss)
        visualize_epochs(
            epoch_train_losses, epoch_dev_loss, f"mlp_lr_{lr_name}_loss.png"
        )

    visualize_configs(
        all_emb_epoch_dev_accs,
        lrs_names,
        "Accuracy",
        "./all_mlp_learning_rates_acc.png",
    )
    visualize_configs(
        all_emb_epoch_dev_losses, lrs_names, "Loss", "./all_mlp_learning_rates_loss.png"
    )
    lrs_names = [str(lr) for lr in lrs]  # for visualization


if __name__ == "__main__":
    # Load raw data for mlp
    # uncomment the following line to run
    dev_data, train_data, test_data = load_data_mlp()

    # load pre-trained embeddings
    # uncomment the following lines to run
    print(f"{'-' * 10} Load Pre-trained Embeddings: {EMBEDDING_TYPE} {'-' * 10}")
    pretrained_embeddings = gensim.downloader.load(EMBEDDING_TYPE)

    # Explore different hidden dimensions
    # uncomment the following line to run
    explore_mlp_structures(dev_data, train_data, test_data, pretrained_embeddings)

    # Explore different activations
    # uncomment the following line to run
    explore_mlp_activations(dev_data, train_data, test_data, pretrained_embeddings)

    # Explore different learning rates
    # uncomment the following line to run
    explore_mlp_learning_rates(dev_data, train_data, test_data, pretrained_embeddings)
