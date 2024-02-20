import torch
from datasets import load_dataset
from gpt.bpe import BPETokenizer
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

bpe_tokenizer = BPETokenizer()
PAD_ID = len(bpe_tokenizer.encoder.encoder) - 1
VOCAB_SIZE = PAD_ID + 1

def load_data():
    print(f"{'-' * 10} Load Dataset {'-' * 10}")
    train_dataset = load_dataset(path="wikitext", name="wikitext-103-raw-v1", split="train")
    dev_dataset = load_dataset(path="wikitext", name="wikitext-103-raw-v1", split="validation")

    return train_dataset, dev_dataset

def tokenize_data(raw_data, tokenizer):
    tokenized_data = [tokenizer(x).view(-1) for x in tqdm(raw_data['text'])]
    return tokenized_data

def prepare_data(tokenized_data, block_size):
    preprocessed_data = []
    for example in tqdm(tokenized_data):
        # drop 0 & 1 length examples
        if len(example) <= 1:
            continue
        # split the text into chunks of block_size
        full_len = len(example)
        if full_len <= block_size:
            preprocessed_data.append(example)
        else:
            for i in range(0, full_len, block_size):
                chunk = example[i:i + block_size]
                if len(chunk) <= 1:
                    continue
                preprocessed_data.append(chunk)
    return preprocessed_data


class WikiTextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx])


def create_datasets(train_data, dev_data, block_size):
    train_tokenized = tokenize_data(train_data, bpe_tokenizer)
    dev_tokenized = tokenize_data(dev_data, bpe_tokenizer)

    train_preprocessed = prepare_data(train_tokenized, block_size)
    dev_preprocessed = prepare_data(dev_tokenized, block_size)

    train_dataset = WikiTextDataset(train_preprocessed, block_size)
    dev_dataset = WikiTextDataset(dev_preprocessed, block_size)

    return train_dataset, dev_dataset

def collate_fn(batch):
    # pad batch ids
    batch_ids = pad_sequence(batch, batch_first=True, padding_value=PAD_ID)
    labels = batch_ids.clone()

    # create mask
    mask = torch.ones(batch_ids.shape, dtype=torch.float32)
    mask[batch_ids == PAD_ID] = 0.0
    return batch_ids, labels, mask


def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
