import easydict
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Union
from easydict import EasyDict

random.seed(42)
torch.manual_seed(42)
nltk.download('punkt')


def load_data() -> Tuple[Dict[str, List[Union[int, str]]], Dict[str, List[Union[int, str]]], Dict[str, List[Union[int, str]]]]:
    """Loads and shuffles the IMDB dataset, then splits it into train, dev, and test sets.
    
    Returns:
        Tuple[Dict, Dict, Dict]: A tuple containing the dev, train, and test datasets.
    """
    print(f"{'-' * 10} Load Dataset {'-' * 10}")
    dataset = load_dataset("imdb")
    dataset = dataset.shuffle()
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    print(f"{'-' * 10} an example from the train set {'-' * 10}")
    print(train_dataset[0])

    print(f"{'-' * 10} an example from the test set {'-' * 10}")
    print(test_dataset[0])

    dev_dataset = train_dataset[:1000]
    train_dataset = train_dataset[1000:21000]
    test_dataset = test_dataset[:1000]

    return dev_dataset, train_dataset, test_dataset


def featurize(sentence: str, embeddings: gensim.models.keyedvectors.KeyedVectors) -> Union[None, torch.FloatTensor]:
    """Converts a sentence into a feature vector by averaging its word embeddings.
    
    Args:
        sentence (str): The input sentence.
        embeddings (gensim.models.keyedvectors.KeyedVectors): The pre-trained word embeddings.
    
    Returns:
        Union[None, torch.FloatTensor]: The average embedding vector of the sentence, or None if no words are found in the embeddings.
    """
    vectors = []
    for word in word_tokenize(sentence.lower()):
        try:
            vectors.append(embeddings[word])
        except KeyError:
            continue

    if not vectors:
        return None
    else:
        avg_vector = np.mean(vectors, axis=0)
        return torch.FloatTensor(avg_vector)


def create_tensor_dataset(raw_data: Dict[str, List[Union[int, str]]],
                          embeddings: gensim.models.keyedvectors.KeyedVectors) -> TensorDataset:
    """Creates a tensor dataset from raw data and embeddings.
    
    Args:
        raw_data (Dict): The raw data containing 'text' and 'label' keys.
        embeddings (gensim.models.keyedvectors.KeyedVectors): The pre-trained word embeddings.
    
    Returns:
        TensorDataset: A tensor dataset containing features and labels.
    """
    all_features, all_labels = [], []
    for text, label in tqdm(zip(raw_data['text'], raw_data['label'])):
        feature = featurize(text, embeddings)
        if feature is not None:
            all_features.append(feature)
            all_labels.append(label)

    features_tensor = torch.stack(all_features)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    return TensorDataset(features_tensor, labels_tensor)


def create_dataloader(dataset: TensorDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Creates a DataLoader from a TensorDataset.
    
    Args:
        dataset (TensorDataset): The tensor dataset.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data every epoch.
    
    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class SentimentClassifier(nn.Module):
    """A simple sentiment classifier using a linear layer and softmax."""
    
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(embed_dim, num_classes)
        self.loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, inp):
        logits = self.fc(inp)
        return logits

    @staticmethod
    def softmax(logits):
        max_logits = torch.max(logits, dim=1, keepdim=True)[0]
        exps = torch.exp(logits - max_logits)
        sum_exps = torch.sum(exps, dim=1, keepdim=True)
        return exps / sum_exps

    def gradient_loss(self, inp, logits, labels):
        bsz = inp.shape[0]
        y_true = torch.zeros(bsz, self.num_classes)
        y_true[torch.arange(bsz), labels] = 1
        probs = self.softmax(logits)
        loss = -torch.sum(y_true * torch.log(probs + 1e-12)) / bsz
        grads_weights = torch.matmul((probs - y_true).T, inp) / bsz
        grads_bias = torch.sum(probs - y_true, axis=0) / bsz
        return grads_weights, grads_bias, loss


def test_softmax():
    """Tests the softmax implementation."""
    test_inp1 = torch.FloatTensor([[1, 2], [1001, 1002]])
    test_inp2 = torch.FloatTensor([[3, 5], [-2003, -2005]])
    assert torch.allclose(SentimentClassifier.softmax(test_inp1),
                          torch.FloatTensor([[0.26894143, 0.73105860], [0.26894143, 0.73105860]]))
    assert torch.allclose(SentimentClassifier.softmax(test_inp2),
                          torch.FloatTensor([[0.11920292, 0.88079703], [0.88079703, 0.11920292]]))


def test_gradient_loss(model: SentimentClassifier):
    """Tests the gradient loss implementation."""
    test_inp1 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    test_logits1 = torch.FloatTensor([[0.3, -0.5], [-0.4, 0.6]])
    test_labels1 = torch.LongTensor([1, 1])
    gw1, gb1, loss1 = model.gradient_loss(test_inp1, test_logits1, test_labels1)
    assert torch.allclose(gw1, torch.FloatTensor([[ 0.8829,  1.3623,  1.8418], [-0.8829, -1.3623, -1.8418]]), atol=1e-4)
    assert torch.allclose(gb1, torch.FloatTensor([ 0.4795, -0.4795]), atol=1e-4)
    assert torch.abs(loss1 - 0.7422) < 1e-4


def accuracy(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
    """Calculates the accuracy of the predictions.
    
    Args:
        logits (torch.FloatTensor): The predicted logits.
        labels (torch.LongTensor): The true labels.
    
    Returns:
        torch.FloatTensor: The accuracy of the predictions.
    """
    assert logits.shape[0] == labels.shape[0]
    preds = torch.argmax(logits, dim=1)
    correct = preds.eq(labels)
    return correct


def evaluate(model: SentimentClassifier, eval_dataloader: DataLoader) -> Tuple[float, float]:
    """Evaluates the model on the evaluation dataloader.
    
    Args:
        model (SentimentClassifier): The model to evaluate.
        eval_dataloader (DataLoader): The dataloader for evaluation.
    
    Returns:
        Tuple[float, float]: The average loss and accuracy on the evaluation set.
    """
    model.eval()
    eval_losses = []
    eval_accs = []
    for batch in tqdm(eval_dataloader):
        inp, labels = batch
        logits = model(inp)
        _, _, loss = model.gradient_loss(inp, logits, labels)
        eval_losses.append(loss.item())
        eval_accs += accuracy(logits, labels).tolist()

    eval_loss, eval_acc = np.array(eval_losses).mean(), np.array(eval_accs).mean()
    print(f"Eval Loss: {eval_loss} Eval Acc: {eval_acc}")
    return eval_loss, eval_acc


def train(model: SentimentClassifier,
          learning_rate: float,
          train_dataloader: DataLoader,
          dev_dataloader: DataLoader,
          num_epochs: int,
          save_path: Union[str, None]=None) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Trains the model and evaluates it on the development set after each epoch.
    
    Args:
        model (SentimentClassifier): The model to train.
        learning_rate (float): The learning rate for gradient descent.
        train_dataloader (DataLoader): The dataloader for training data.
        dev_dataloader (DataLoader): The dataloader for development data.
        num_epochs (int): The number of epochs to train for.
        save_path (Union[str, None]): The path to save the model to.
    
    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: Training and dev losses and accuracies for each epoch.
    """
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
            inp, labels = batch
            logits = model(inp)
            grads_weights, grads_bias, loss = model.gradient_loss(inp, logits, labels)
            with torch.no_grad():
                model.fc.weight -= learning_rate * grads_weights
                model.fc.bias -= learning_rate * grads_bias

            train_losses.append(loss.item())
            train_accs += accuracy(logits, labels).tolist()

        all_epoch_train_losses.append(np.array(train_losses).mean())
        all_epoch_train_accs.append(np.array(train_accs).mean())

        print(f"{'-' * 10} Epoch {epoch}: Evaluation on Dev Set {'-' * 10}")
        dev_loss, dev_acc = evaluate(model, dev_dataloader)
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
    """Visualizes the training and development losses over the epochs.
    
    Args:
        epoch_train_losses (List[float]): The training losses for each epoch.
        epoch_dev_losses (List[float]): The development losses for each epoch.
        save_fig_path (str): The path to save the figure to.
    """
    plt.clf()
    plt.plot(epoch_train_losses, label='train')
    plt.plot(epoch_dev_losses, label='dev')
    plt.xticks(np.arange(0, len(epoch_train_losses), 5).astype(np.int32))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_fig_path)


def run_grad_descent(config: EasyDict,
                     dev_data: Dict[str, List[Union[int, str]]],
                     train_data: Dict[str, List[Union[int, str]]],
                     test_data: Dict[str, List[Union[int, str]]]) -> Tuple[List[float], List[float], List[float], List[float], float, float]:
    """Runs gradient descent using the provided configuration and data.
    
    Args:
        config (EasyDict): The configuration for the run.
        dev_data (Dict): The development data.
        train_data (Dict): The training data.
        test_data (Dict): The testing data.
    
    Returns:
        Tuple[List[float], List[float], List[float], List[float], float, float]: The training and development losses and accuracies for each epoch, and the test loss and accuracy.
    """
    print(f"{'-' * 10} Load Pre-trained Embeddings: {config.embeddings} {'-' * 10}")
    embeddings = gensim.downloader.load(config.embeddings)

    print(f"{'-' * 10} Create Datasets {'-' * 10}")
    train_dataset = create_tensor_dataset(train_data, embeddings)
    dev_dataset = create_tensor_dataset(dev_data, embeddings)
    test_dataset = create_tensor_dataset(test_data, embeddings)

    print(f"{'-' * 10} Create Dataloaders {'-' * 10}")
    train_dataloader = create_dataloader(train_dataset, config.batch_size, shuffle=True)
    dev_dataloader = create_dataloader(dev_dataset, config.batch_size, shuffle=False)
    test_dataloader = create_dataloader(test_dataset, config.batch_size, shuffle=False)

    print(f"{'-' * 10} Load Model {'-' * 10}")
    model = SentimentClassifier(embeddings.vector_size, config.num_classes)

    print(f"{'-' * 10} Test Softmax {'-' * 10}")
    test_softmax()
    print(f"{'-' * 10} Pass Softmax Test {'-' * 10}")

    print(f"{'-' * 10} Test Backward {'-' * 10}")
    test_gradient_loss(model)
    print(f"{'-' * 10} Pass Backward Test {'-' * 10}")

    print(f"{'-' * 10} Start Training {'-' * 10}")
    all_epoch_train_losses, all_epoch_train_accs, all_epoch_dev_losses, all_epoch_dev_accs = (
        train(model, config.lr, train_dataloader, dev_dataloader, config.num_epochs, config.save_path))
    model.load_state_dict(torch.load(config.save_path))

    print(f"{'-' * 10} Evaluate on Test Set {'-' * 10}")
    test_loss, test_acc = evaluate(model, test_dataloader)

    return all_epoch_train_losses, all_epoch_train_accs, all_epoch_dev_losses, all_epoch_dev_accs, test_loss, test_acc

