import easydict
import nltk
from nltk.tokenize import word_tokenize  # for tokenization
import numpy as np  # for numerical operators
import matplotlib.pyplot as plt  # for plotting
import gensim.downloader  # for download word embeddings
import torch
import torch.nn as nn
import random
from tqdm import tqdm  # progress bar
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Union

# set random seeds
random.seed(42)
torch.manual_seed(42)

nltk.download('punkt')


def load_data_mlp() -> Tuple[
    Dict[str, List[Union[int, str]]], Dict[str, List[Union[int, str]]], Dict[str, List[Union[int, str]]]]:
    # download dataset
    print(f"{'-' * 10} Load Dataset {'-' * 10}")
    dataset = load_dataset("imdb")
    dataset = dataset.shuffle()  # shuffle the data
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    print(f"{'-' * 10} an example from the train set {'-' * 10}")
    print(train_dataset[0])

    print(f"{'-' * 10} an example from the test set {'-' * 10}")
    print(test_dataset[0])

    # cap the train and test sets at 20k and 10k, use the first 1k examples from the train set as the dev set
    # we convert each split into a torch dataset
    dev_dataset = train_dataset[:5000]
    train_dataset = train_dataset[5000:25000]
    test_dataset = test_dataset[:10000]

    return dev_dataset, train_dataset, test_dataset

def featurize(sentence: str, embeddings: gensim.models.keyedvectors.KeyedVectors) -> Union[None, torch.FloatTensor]:
    # sequence of word embeddings
    vectors = []

    # map each word to its embedding
    for word in word_tokenize(sentence.lower()):
        try:
            vectors.append(embeddings[word])
        except KeyError:
            pass

    # TODO (Copy from your HW1): complete the function to compute the average embedding of the sentence
    # your return should be
    # None - if the vector sequence is empty, i.e. the sentence is empty or None of the words in the sentence is in the embedding vocabulary
    # A torch tensor of shape (embed_dim,) - the average word embedding of the sentence
    # Hint: follow the hints in the pdf description
    raise NotImplementedError


def create_tensor_dataset(raw_data: Dict[str, List[Union[int, str]]],
                          embeddings: gensim.models.keyedvectors.KeyedVectors) -> TensorDataset:
    all_features, all_labels = [], []
    for text, label in tqdm(zip(raw_data['text'], raw_data['label'])):

        # TODO (Copy from your HW1): complete the for loop to featurize each sentence
        # only add the feature and label to the list if the feature is not None
        raise NotImplementedError
        # your code ends here

    # stack all features and labels into two single tensors and create a TensorDataset
    features_tensor = torch.stack(all_features)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    return TensorDataset(features_tensor, labels_tensor)

def create_dataloader(dataset: TensorDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class SentimentClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, hidden_dims: List[int], activation: str = 'Sigmoid'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # TODO: define the activation functions
        # activation function
        if activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError

        # linear layers for the MLP
        self.linears = nn.ModuleList()
        # TODO: define the MLP given the hidden dimensions
        # - embed_dim is the dimension of the word embeddings (input to the MLP)
        # - num_classes is the number of classes (output of the MLP)
        # - hidden_dims is a list of integers, where each integer is the dimension of a hidden layer in the MLP
        # - The linear layers should have the following dimensions:
        #       [embed_dim, hidden_dims[0]], [hidden_dims[0], hidden_dims[1]], ..., [hidden_dims[-1], num_classes]
        # Hint:
        # - Remember to consider the case when there are no hidden layers (i.e. hidden_dims is an empty list)
        #       in this case, it essentially degrades to the architecture we used in hw 1
        raise NotImplementedError
        # your code ends here

        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inp):

        # TODO: complete the forward function
        # Hint remember to apply the activation function to all hidden layers except the last one
        raise NotImplementedError
        # your code ends here

        return logits


def accuracy(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
    assert logits.shape[0] == labels.shape[0]
    # TODO (Copy from your HW1): complete the function to compute the accuracy
    # Hint: follow the hints in the pdf description, the return should be a tensor of 0s and 1s with the same shape as labels
    # labels is a tensor of shape (batch_size,)
    # logits is a tensor of shape (batch_size, num_classes)
    raise NotImplementedError


def evaluate(model: SentimentClassifier, eval_dataloader: DataLoader) -> Tuple[float, float]:
    model.eval()
    eval_losses = []
    eval_accs = []
    for batch in tqdm(eval_dataloader):
        inp, labels = batch
        # forward pass
        logits = model(inp)
        # loss and accuracy computation
        loss = model.loss(logits, labels)
        eval_losses.append(loss.item())
        eval_accs += accuracy(logits, labels).tolist()

    eval_loss, eval_acc = np.array(eval_losses).mean(), np.array(eval_accs).mean()
    print(f"Eval Loss: {eval_loss} Eval Acc: {eval_acc}")
    return eval_loss, eval_acc


def train(model: SentimentClassifier,
          optimizer: torch.optim.Optimizer,
          train_dataloader: DataLoader,
          dev_dataloader: DataLoader,
          num_epochs: int,
          save_path: Union[str, None] = None):
    # record the training process and model performance for each epoch
    all_epoch_train_losses = []
    all_epoch_train_accs = []
    all_epoch_dev_losses = []
    all_epoch_dev_accs = []
    best_acc = -1.
    for epoch in range(num_epochs):
        model.train()
        print(f"{'-' * 10} Epoch {epoch}: Training {'-' * 10}")
        train_losses = []
        train_accs = []
        for batch in tqdm(train_dataloader):
            # zero the gradient history
            # will explain more about it in future lectures and homework
            optimizer.zero_grad()
            inp, labels = batch
            # forward pass
            logits = model(inp)
            # compute loss and backpropagate
            # will explain more about it in future lectures and homework
            loss = model.loss(logits, labels)
            loss.backward()
            optimizer.step()
            # record the loss and accuracy
            train_losses.append(loss.item())
            train_accs += accuracy(logits, labels).tolist()

        all_epoch_train_losses.append(np.array(train_losses).mean())
        all_epoch_train_accs.append(np.array(train_accs).mean())

        # evaluate on the dev set
        print(f"{'-' * 10} Epoch {epoch}: Evaluation on Dev Set {'-' * 10}")
        dev_loss, dev_acc = evaluate(model, dev_dataloader)
        # save the model if it achieves better accuracy on the dev set
        if dev_acc > best_acc:
            best_acc = dev_acc
            print(f"{'-' * 10} Epoch {epoch}: Best Acc so far: {best_acc} {'-' * 10}")
            if save_path:
                print(f"{'-' * 10} Epoch {epoch}: Saving model to {save_path} {'-' * 10}")
                torch.save(model.state_dict(), save_path)

        all_epoch_dev_losses.append(dev_loss)
        all_epoch_dev_accs.append(dev_acc)

    return all_epoch_train_losses, all_epoch_train_accs, all_epoch_dev_losses, all_epoch_dev_accs


def visualize_epochs(epoch_train_losses: List[float], epoch_dev_losses: List[float], save_fig_path: str):
    plt.clf()
    ax = plt.gca()
    ax.set_ylim([0.4, 0.7])
    plt.plot(epoch_train_losses, label='train')
    plt.plot(epoch_dev_losses, label='dev')
    plt.xticks(np.arange(0, len(epoch_train_losses)).astype(np.int32)),
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_fig_path)


def visualize_configs(all_config_epoch_stats: List[List[float]], config_names: List[str], metric_name: str,
                      save_fig_path: str):
    plt.clf()
    ax = plt.gca()
    if metric_name == "Accuracy":
        ax.set_ylim([0.65, 0.8])
    if metric_name == "Loss":
        ax.set_ylim([0.4, 0.65])
    for config_epoch_stats, config_name in zip(all_config_epoch_stats, config_names):
        plt.plot(config_epoch_stats, label=config_name)
    plt.xticks(np.arange(0, len(all_config_epoch_stats[0])).astype(np.int32)),
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(save_fig_path)


def run_mlp(config: easydict.EasyDict,
            embeddings: gensim.models.keyedvectors.KeyedVectors,
            dev_dataset: TensorDataset,
            train_dataset: TensorDataset,
            test_dataset: TensorDataset):

    print(f"{'-' * 10} Create Dataloaders {'-' * 10}")
    train_dataloader = create_dataloader(train_dataset, config.batch_size, shuffle=True)
    dev_dataloader = create_dataloader(dev_dataset, config.batch_size, shuffle=False)
    test_dataloader = create_dataloader(test_dataset, config.batch_size, shuffle=False)

    print(f"{'-' * 10} Load Model {'-' * 10}")
    model = SentimentClassifier(embeddings.vector_size, config.num_classes, config.hidden_dims, config.activation)
    # define optimizer that manages the model's parameters and gradient updates
    # we will learn more about optimizers in future lectures and homework
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    print(f"{'-' * 10} Start Training {'-' * 10}")
    all_epoch_train_losses, all_epoch_train_accs, all_epoch_dev_losses, all_epoch_dev_accs = (
        train(model, optimizer, train_dataloader, dev_dataloader, config.num_epochs, config.save_path))
    model.load_state_dict(torch.load(config.save_path))

    print(f"{'-' * 10} Evaluate on Test Set {'-' * 10}")
    test_loss, test_acc = evaluate(model, test_dataloader)

    return all_epoch_train_losses, all_epoch_train_accs, all_epoch_dev_losses, all_epoch_dev_accs, test_loss, test_acc
