import easydict
import nltk
from nltk.tokenize import word_tokenize  # for tokenization
import numpy as np  # for numerical operators
import matplotlib.pyplot as plt  # for plotting
import gensim.downloader  # for downloading word embeddings
import torch
import torch.nn as nn
import random
from tqdm import tqdm  # progress bar
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Union

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
nltk.download("punkt")

def load_data_mlp() -> Tuple[
    Dict[str, List[Union[int, str]]],
    Dict[str, List[Union[int, str]]],
    Dict[str, List[Union[int, str]]],
]:
    """
    Load and shuffle the IMDB dataset. Split it into training, development, and test sets.

    Returns:
        Tuple containing the development, training, and test datasets.
    """
    # Download and shuffle the dataset
    print(f"{'-' * 10} Load Dataset {'-' * 10}")
    dataset = load_dataset("imdb")
    dataset = dataset.shuffle()  # Shuffle the data
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Display examples from the dataset
    print(f"{'-' * 10} an example from the train set {'-' * 10}")
    print(train_dataset[0])
    print(f"{'-' * 10} an example from the test set {'-' * 10}")
    print(test_dataset[0])

    # Split datasets and cap the sizes
    dev_dataset = train_dataset[:5000]
    train_dataset = train_dataset[5000:25000]
    test_dataset = test_dataset[:10000]

    return dev_dataset, train_dataset, test_dataset

def featurize(
    sentence: str, embeddings: gensim.models.keyedvectors.KeyedVectors
) -> Union[None, torch.FloatTensor]:
    """
    Convert a sentence into an average of its word embeddings.

    Parameters:
        sentence: The sentence to be featurized.
        embeddings: Pre-loaded word embeddings.

    Returns:
        A tensor representing the average word embedding of the sentence or None if the sentence has no known words.
    """
    vectors = []
    for word in word_tokenize(sentence.lower()):
        try:
            vectors.append(embeddings[word])
        except KeyError:
            pass

    if len(vectors) == 0:
        return None
    else:
        avg_vector = np.mean(vectors, axis=0)
        return torch.FloatTensor(avg_vector)

def create_tensor_dataset(
    raw_data: Dict[str, List[Union[int, str]]],
    embeddings: gensim.models.keyedvectors.KeyedVectors,
) -> TensorDataset:
    """
    Create a TensorDataset from raw data using pre-trained embeddings.

    Parameters:
        raw_data: A dictionary containing 'text' and 'label' lists.
        embeddings: Pre-loaded word embeddings.

    Returns:
        A TensorDataset containing features and labels.
    """
    all_features, all_labels = [], []
    for text, label in tqdm(zip(raw_data["text"], raw_data["label"])):
        feature = featurize(text, embeddings)
        if feature is not None:
            all_features.append(feature)
            all_labels.append(label)

    features_tensor = torch.stack(all_features)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    return TensorDataset(features_tensor, labels_tensor)

def create_dataloader(
    dataset: TensorDataset, batch_size: int, shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader from a TensorDataset.

    Parameters:
        dataset: The dataset to be loaded.
        batch_size: The size of each batch.
        shuffle: Whether to shuffle the dataset.

    Returns:
        A DataLoader for the given dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class SentimentClassifier(nn.Module):
    """
    A simple MLP for sentiment classification.
    """
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dims: List[int],
        activation: str = "Sigmoid",
    ):
        """
        Initialize the SentimentClassifier model.

        Parameters:
            embed_dim: Dimensionality of the input features.
            num_classes: Number of classes to predict.
            hidden_dims: A list of integers specifying the size of each hidden layer.
            activation: The activation function to use ('Sigmoid', 'Tanh', 'ReLU', 'GeLU').
        """
        super().__init__()
        self.embed_dim = embed_dim


        self.num_classes = num_classes

        # Define the activation function
        activations = {
            "Sigmoid": nn.Sigmoid(),
            "Tanh": nn.Tanh(),
            "ReLU": nn.ReLU(),
            "GeLU": nn.GELU(),
        }
        self.activation = activations.get(activation)
        if self.activation is None:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Initialize linear layers
        self.linears = nn.ModuleList()
        if not hidden_dims:
            self.linears.append(nn.Linear(embed_dim, num_classes))
        else:
            prev_dim = embed_dim
            for hidden_dim in hidden_dims:
                self.linears.append(nn.Linear(prev_dim, hidden_dim))
                prev_dim = hidden_dim
            self.linears.append(nn.Linear(hidden_dims[-1], num_classes))

        self.loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, inp):
        """
        Forward pass of the model.

        Parameters:
            inp: Input tensor.

        Returns:
            The logits predicted by the model.
        """
        for linear in self.linears[:-1]:
            inp = self.activation(linear(inp))
        logits = self.linears[-1](inp)
        return logits

def accuracy(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
    """
    Compute the accuracy of predictions.

    Parameters:
        logits: The logits returned by the model.
        labels: The true labels.

    Returns:
        The accuracy of the predictions.
    """
    preds = torch.argmax(logits, dim=1)
    correct = preds.eq(labels)
    return correct

def evaluate(
    model: SentimentClassifier, eval_dataloader: DataLoader
) -> Tuple[float, float]:
    """
    Evaluate the model on a given dataset.

    Parameters:
        model: The SentimentClassifier to evaluate.
        eval_dataloader: The DataLoader for the evaluation dataset.

    Returns:
        A tuple containing the average loss and accuracy.
    """
    model.eval()
    eval_losses = []
    eval_accs = []
    for batch in tqdm(eval_dataloader):
        inp, labels = batch
        logits = model(inp)
        loss = model.loss(logits, labels)
        eval_losses.append(loss.item())
        eval_accs += accuracy(logits, labels).tolist()

    eval_loss, eval_acc = np.mean(eval_losses), np.mean(eval_accs)
    print(f"Eval Loss: {eval_loss} Eval Acc: {eval_acc}")
    return eval_loss, eval_acc

def train(
    model: SentimentClassifier,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    num_epochs: int,
    save_path: Union[str, None] = None,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Train the SentimentClassifier model.

    Parameters:
        model: The SentimentClassifier to train.
        optimizer: The optimizer to use for training.
        train_dataloader: The DataLoader for the training dataset.
        dev_dataloader: The DataLoader for the development dataset.
        num_epochs: The number of epochs to train for.
        save_path: Path to save the best model.

    Returns:
        A tuple containing lists of training and development losses and accuracies across epochs.
    """
    best_acc = -1.0
    for epoch in range(num_epochs):
        model.train()
        print(f"{'-' * 10} Epoch {epoch}: Training {'-' * 10}")
        train_losses, train_accs = [], []
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            inp, labels = batch
            logits = model(inp)
            loss = model.loss(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_accs += accuracy(logits, labels).tolist()

        dev_loss, dev_acc = evaluate(model, dev_dataloader)
        if dev_acc > best_acc:
            best_acc = dev_acc
            print(f"{'-' * 10} Epoch {epoch}: Best Acc so far: {best_acc} {'-' * 10}")
            if save_path:
                torch.save(model.state_dict(), save_path)

    return train_losses, train_accs, [dev_loss], [dev_acc]

def visualize_epochs(
    epoch_train_losses: List[float], epoch_dev_losses: List[float], save_fig_path: str
):
    """
    Visualize training and development losses over epochs.

    Parameters:
        epoch_train_losses: List of training losses per epoch.
        epoch_dev_losses: List of development losses per epoch.
        save_fig_path: Path to save the resulting plot.
    """
    plt.plot(epoch_train_losses, label="train")
    plt.plot(epoch_dev_losses, label="dev")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_fig_path)

def visualize_configs(
    all_config_epoch

_stats: List[List[float]],
    config_names: List[str],
    metric_name: str,
    save_fig_path: str,
):
    """
    Visualize the performance of different configurations.

    Parameters:
        all_config_epoch_stats: A list of performance metrics for each configuration.
        config_names: Names of the configurations.
        metric_name: The name of the metric (e.g., "Accuracy" or "Loss").
        save_fig_path: Path to save the resulting plot.
    """
    for config_epoch_stats, config_name in zip(all_config_epoch_stats, config_names):
        plt.plot(config_epoch_stats, label=config_name)
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(save_fig_path)

def run_mlp(
    config: easydict.EasyDict,
    embeddings: gensim.models.keyedvectors.KeyedVectors,
    dev_dataset: TensorDataset,
    train_dataset: TensorDataset,
    test_dataset: TensorDataset,
) -> Tuple[List[float], List[float], List[float], List[float], float, float]:
    """
    Run the entire MLP pipeline with the given configuration and datasets.

    Parameters:
        config: Configuration parameters as an easydict.
        embeddings: Pre-loaded word embeddings.
        dev_dataset: The development dataset as a TensorDataset.
        train_dataset: The training dataset as a TensorDataset.
        test_dataset: The test dataset as a TensorDataset.

    Returns:
        A tuple containing training losses, training accuracies, development losses, development accuracies, test loss, and test accuracy.
    """
    train_dataloader = create_dataloader(train_dataset, config.batch_size, shuffle=True)
    dev_dataloader = create_dataloader(dev_dataset, config.batch_size, shuffle=False)
    test_dataloader = create_dataloader(test_dataset, config.batch_size, shuffle=False)

    model = SentimentClassifier(
        embeddings.vector_size,
        config.num_classes,
        config.hidden_dims,
        config.activation,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    (
        all_epoch_train_losses,
        all_epoch_train_accs,
        all_epoch_dev_losses,
        all_epoch_dev_accs,
    ) = train(
        model,
        optimizer,
        train_dataloader,
        dev_dataloader,
        config.num_epochs,
        config.save_path,
    )
    model.load_state_dict(torch.load(config.save_path))

    test_loss, test_acc = evaluate(model, test_dataloader)

    return (
        all_epoch_train_losses,
        all_epoch_train_accs,
        all_epoch_dev_losses,
        all_epoch_dev_accs,
        test_loss,
        test_acc,
    )