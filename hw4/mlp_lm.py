import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from sentence_splitter import SentenceSplitter
from transformers import AutoTokenizer, TopPLogitsWarper
from collections import Counter, defaultdict  # we will using "Counter" data structure for counting word co-occurences
from typing import List

torch.manual_seed(42)
random.seed(42)


def load_data_mlp_lm():
    print(f"{'-' * 10} Load Dataset {'-' * 10}")
    train_dataset = load_dataset(path="wikitext", name="wikitext-103-raw-v1", split="train")
    dev_dataset = load_dataset(path="wikitext", name="wikitext-103-raw-v1", split="validation")

    print(f"{'-' * 10} an example from the train set {'-' * 10}")
    print(train_dataset['text'][10])

    return train_dataset, dev_dataset


def preprocess_data(data, local_window_size, splitter, tokenizer):
    x_data = []
    y_data = []
    for paragraph in tqdm(data['text']):

        # if the paragraph is too short, skip it
        if len(paragraph) < 5:
            continue

        # iterate over sentences given by our sentence splitter
        for sentence in splitter.split(paragraph):

            # tokenize the words in the our sentence
            tokens = tokenizer.tokenize(sentence)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            # drop short sentences
            if len(tokens) < local_window_size + 1:
                break

            for idx, _ in enumerate(token_ids):

                if idx + local_window_size >= len(tokens):
                    # have already traversed all of the sentence
                    break

                # TODO: Select a subset of token_ids from idx -> idx + local_window_size as input and put it to x
                # Select a subset of token_ids from idx -> idx + local_window_size as input and put it to x: list of context token_ids
                # Then select the word immediately after this window as output and put it to y: the target next token_id
                raise NotImplementedError
                # your code ends here

                x_data.append(x)
                y_data.append(y)

    # making tensors
    x_data = torch.LongTensor(x_data)
    y_data = torch.LongTensor(y_data)

    # creating a dataset
    return TensorDataset(x_data, y_data)


class NPLMFirstBlock(nn.Module):

    def __init__(self, vocab_size, embed_dim, local_window_size, hidden_dim, dropout_p):
        super(NPLMFirstBlock, self).__init__()
        self.local_window_size = local_window_size  # size of the context window
        self.embed_dim = embed_dim  # dimension of the word embedding vectors
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(local_window_size * embed_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs):
        # TODO: implement the forward pass
        raise NotImplementedError
        # looking up the word embeddings from self.embeddings()
        # And concatenating them
        # Note this is done for a batch of instances.


        # Transform embeddings with a linear layer and tanh activation


        # apply layer normalization


        # apply dropout

        # your code ends here

        return final_embeds


class NPLMBlock(nn.Module):

    def __init__(self, hidden_dim, dropout_p):
        super(NPLMBlock, self).__init__()
        self.hidden_dim = hidden_dim  # size of the hidden representation
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs):
        # TODO: implement the forward pass
        raise NotImplementedError
        # apply linear transformation and tanh activation


        # add residual connection


        # apply layer normalization


        # apply dropout

        # your code ends here

        return final_inputs


class NPLMFinalBlock(nn.Module):

    def __init__(self, vocab_size, hidden_dim):
        super(NPLMFinalBlock, self).__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, inputs):
        # TODO: implement the forward pass
        raise NotImplementedError
        # apply linear transformation

        # apply log_softmax to get log-probabilities (logits)

        # your code ends here

        return log_probs


class NPLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, local_window_size, hidden_dim, num_blocks, dropout_p):
        super(NPLM, self).__init__()

        self.first_layer = NPLMFirstBlock(vocab_size, embed_dim, local_window_size, hidden_dim, dropout_p)

        self.intermediate_layers = nn.ModuleList()

        # TODO: create num_blocks of NPLMBlock as intermediate layers
        # append them to self.intermediate_layers
        raise NotImplementedError

        # your code ends here

        self.final_layer = NPLMFinalBlock(vocab_size, hidden_dim)

    def forward(self, inputs):
        # TODO: implement the forward pass
        raise NotImplementedError
        # input layer

        # multiple middle layers
        # remember to apply the ReLU activation function after each layer

        # output layer

        # your code ends here

        return log_probs


def test_model_forward(model, sample, local_window_size, tokenizer):
    inp, target = sample

    log_probs = model(inp)

    print(f"Batched inputs: {inp}")
    print(f"Batched outputs: {target}")
    print(f"Model outpus: {log_probs}")

    # the first dimension of the output should correspond to the batch size
    assert log_probs.shape[0] == target.shape[0]

    # the other dimension of the output should equal the vocab size since the model
    # produces a distribution over the vocabulary
    assert log_probs.shape[1] == tokenizer.vocab_size


def train(model, train_dataloader, dev_dataloader, criterion, optimizer, scheduler, num_epochs, save_path,
          print_every=100):
    best_ppl = np.inf
    all_epoch_train_losses = []
    all_epoch_train_ppls = []
    all_epoch_dev_losses = []
    all_epoch_dev_ppls = []
    for epoch in range(num_epochs):
        train_losses = []
        train_ppls = []
        print(f"{'-' * 10} Epoch {epoch}: Training {'-' * 10}")
        for idx, batch in tqdm(enumerate(train_dataloader)):
            inp, target = batch

            # get log probabilities over next words
            log_probs = model(inp)

            # compute loss function
            loss = criterion(log_probs, target)

            # TODO extract perplexity
            # remember the connection between perplexity and cross-entropy loss
            # name the perplexity result as 'ppl'
            raise NotImplementedError
            # your code ends here

            # backward pass and update gradient
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            train_ppls.append(ppl.item())

            if idx % print_every == 0:
                learning_rate = optimizer.param_groups[0]['lr']
                print(f"{'-' * 10} Training Iteration {idx} complete. Loss: {loss.item()}; "
                      f"Perplexity: {ppl}; learning rate: {learning_rate} {'-' * 10}")

        all_epoch_train_losses.append(np.array(train_losses).mean())
        all_epoch_train_ppls.append(np.array(train_ppls).mean())

        print(f"{'-' * 10} Epoch {epoch}: Evaluation on Dev Set {'-' * 10}")
        dev_loss, dev_ppl = evaluate(model, dev_dataloader, criterion)
        print(f"Dev Perplexity: {dev_ppl}; Dev Loss: {dev_loss}")
        if dev_ppl < best_ppl:
            print(f"{'-' * 10} Epoch {epoch} Best development perplexity improved from {best_ppl} to {dev_ppl}, "
                  f"saving model... {'-' * 10}")
            best_ppl = dev_ppl
            # saving best model
            torch.save(model.state_dict(), save_path)
        all_epoch_dev_losses.append(dev_loss)
        all_epoch_dev_ppls.append(dev_ppl)

    return all_epoch_train_losses, all_epoch_train_ppls, all_epoch_dev_losses, all_epoch_dev_ppls


def evaluate(model, eval_dataloader, criterion):
    model.eval()

    loss = 0.
    count = 0

    with torch.no_grad():  # turn off gradient calculation because we want to evaluate the model
        for idx, batch in tqdm(enumerate(eval_dataloader)):
            inp, target = batch
            log_probs = model(inp)
            loss += criterion(log_probs, target).item()
            count += 1
    avg_loss = loss / count
    # TODO: compute perplexity
    # name the perplexity result as 'avg_ppl'
    raise NotImplementedError
    # your code ends here
    return avg_loss, avg_ppl


def generate_text(prompt, model, tokenizer, local_window_size, top_p, max_len=30):
    # tokenize the text and turn it into indices
    tokenized_prompt = tokenizer.tokenize(prompt)
    tokenized_prompt_ids = tokenizer.convert_tokens_to_ids(tokenized_prompt)

    count = 0
    while count < max_len:
        # select the tokens in the window
        new_input = torch.tensor(tokenized_prompt_ids[-local_window_size:])

        # compute model output
        output = model(new_input)

        if top_p:
            # filter the top_p generations
            output = TopPLogitsWarper(top_p=top_p)(
                None, output
            )
            probs = F.softmax(output, dim=-1)
            pred = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # greedy decoding
            pred = torch.argmax(output).item()

        tokenized_prompt_ids.append(pred)
        count += 1

    # turn token-ids to their strings
    tokens = tokenizer.convert_ids_to_tokens(tokenized_prompt_ids)

    # turn sub-words into a sentence
    generation = tokenizer.convert_tokens_to_string(tokens)
    print(f"Generated text: {generation}")


def visualize_epochs(epoch_train_stats: List[float], epoch_dev_stats: List[float], stat_name: str, save_fig_path: str):
    plt.clf()
    plt.plot(epoch_train_stats, label='train')
    plt.plot(epoch_dev_stats, label='dev')
    plt.xticks(np.arange(0, len(epoch_train_stats)).astype(np.int32)),
    plt.xlabel('Epochs')
    plt.ylabel(f'{stat_name}')
    plt.legend()
    plt.savefig(save_fig_path)


def run_mlp_lm(config, train_data, dev_data):
    # create sentence splitter and tokenizer
    splitter = SentenceSplitter(language='en')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # preprocess data
    print(f"{'-' * 10} Preprocess Data {'-' * 10}")
    train_dataset = preprocess_data(train_data[-100000:], config.local_window_size, splitter, tokenizer)
    dev_dataset = preprocess_data(dev_data, config.local_window_size, splitter, tokenizer)

    # create dataloaders
    print(f"{'-' * 10} Create Dataloader {'-' * 10}")
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)

    # create model
    print(f"{'-' * 10} Create Model {'-' * 10}")
    model = NPLM(tokenizer.vocab_size, config.embed_dim, config.local_window_size, config.hidden_dim, config.num_blocks,
                 config.dropout_p)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"{'-' * 10} # of parameters: {params} {'-' * 10}")

    # test model forward
    print(f"{'-' * 10} Test Model Forward {'-' * 10}")
    sample = next(iter(train_dataloader))
    test_model_forward(model, sample, config.local_window_size, tokenizer)

    # using ADAM optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # scheduler, for adjusting learning rate
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.decay)

    print(f"{'-' * 10} Start Training {'-' * 10}")
    criterion = nn.NLLLoss()
    return train(model, train_dataloader, dev_dataloader, criterion, optimizer, scheduler, config.num_epochs,
                 config.save_path)


def sample_from_mlp_lm(config, dev_data):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = NPLM(tokenizer.vocab_size, config.embed_dim, config.local_window_size, config.hidden_dim, config.num_blocks,
                 config.dropout_p)
    print(f"{'-' * 10} Load Model Weights {'-' * 10}")
    model.load_state_dict(torch.load(config.save_path, map_location=torch.device('cpu')))
    model.eval()

    print(f"{'-' * 10} Evaluate on the Dev Set {'-' * 10}")
    splitter = SentenceSplitter(language='en')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    dev_dataset = preprocess_data(dev_data, config.local_window_size, splitter, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)
    dev_loss, dev_ppl = evaluate(model, dev_dataloader, criterion=nn.NLLLoss())
    print(f"Dev Perplexity: {dev_ppl}; Dev Loss: {dev_loss}")

    prompt = "The best perks of living on the east"

    print(f"{'-' * 10} Sample From the Model {'-' * 10}")
    # greedy decoding
    print(" {'-' * 10} greedy {'-' * 10}")
    generate_text(prompt, model, tokenizer, config.local_window_size, top_p=None)

    # sampling with p=0.0; note this is equivalent to the greedy search
    print(f"{'-' * 10} sampling with p=0.0 {'-' * 10}")
    generate_text(prompt, model, tokenizer, config.local_window_size, top_p=0.0)

    # sampling with p=0.3
    print(f"{'-' * 10} sampling with p=0.3 {'-' * 10}")
    generate_text(prompt, model, tokenizer, config.local_window_size, top_p=0.3)

    # sampling with p=1.0
    print(f"{'-' * 10} sampling with p=1.0 {'-' * 10}")
    generate_text(prompt, model, tokenizer, config.local_window_size, top_p=1.0)
