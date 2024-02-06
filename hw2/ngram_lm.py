import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from sentence_splitter import SentenceSplitter
from transformers import AutoTokenizer
from collections import Counter, defaultdict


def load_data():
    """Load the wikitext dataset for training and validation.
    
    Returns:
        tuple: A tuple containing the training and validation datasets.
    """
    print(f"{'-' * 10} Load Dataset {'-' * 10}")
    train_dataset = load_dataset(path="wikitext", name="wikitext-103-raw-v1", split="train")
    dev_dataset = load_dataset(path="wikitext", name="wikitext-103-raw-v1", split="validation")

    print(f"{'-' * 10} an example from the train set {'-' * 10}")
    print(train_dataset['text'][10])

    return train_dataset, dev_dataset


def sentence_split_and_tokenize_demo(data, splitter, tokenizer):
    """Demonstrate sentence splitting and tokenization on a dataset example.
    
    Args:
        data: The dataset to use for the demonstration.
        splitter: The sentence splitter.
        tokenizer: The tokenizer.
    """
    print(f"{'-' * 10} split into sentences {'-' * 10}")
    sentences = splitter.split(data['text'][10])
    for sentence in sentences:
        print(f" -> {sentence}")

    print(f"{'-' * 10} tokenize the first sentence {'-' * 10}")
    tokens = tokenizer.tokenize(sentences[0])
    print(f"Tokens: {tokens}")
    print(f"Tokens to indices: {tokenizer.convert_tokens_to_ids(tokens)}")


def create_ngrams(data, n, splitter, tokenizer):
    """Create n-grams from a dataset.
    
    Args:
        data: The dataset to use for creating n-grams.
        n: The number of words in each n-gram.
        splitter: The sentence splitter.
        tokenizer: The tokenizer.
    
    Returns:
        tuple: A tuple containing the predictions of the next word, their scores, and sorted next word candidates.
    """
    ngrams = Counter()
    ngram_context = Counter()
    next_word_candidates = defaultdict(set)

    for paragraph in tqdm(data['text']):
        if len(paragraph) < 3:
            continue

        for sentence in splitter.split(paragraph):
            tokens = tokenizer.tokenize(sentence)
            if len(tokens) < 7:
                break

            for idx in range(len(tokens) - n + 1):
                ngram = tuple(tokens[idx:idx + n])
                context = ngram[:-1]
                next_word = ngram[-1]

                ngrams[ngram] += 1
                ngram_context[context] += 1
                next_word_candidates[context].add(next_word)

    sorted_next_word_candidates = defaultdict(list)
    for context, next_words in next_word_candidates.items():
        sorted_next_word_candidates[context] = sorted(list(next_words))

    next_word_pred = {}
    next_word_scores = {}
    for context, next_words in sorted_next_word_candidates.items():
        scores = [ngrams[context + (nw,)] / ngram_context[context] for nw in next_words]
        next_word_pred[context] = next_words[np.argmax(scores)]
        next_word_scores[context] = np.array(scores)

    return next_word_pred, next_word_scores, sorted_next_word_candidates


def generate_text(word_pred, tokenizer, n, prefix, max_len=100, stop_token="."):
    """Generate text using word predictions.
    
    Args:
        word_pred: Word predictions.
        tokenizer: The tokenizer.
        n: The number of words in each n-gram.
        prefix: The prefix to start the text generation.
        max_len: The maximum length of the generated text.
        stop_token: The token that signifies the end of text generation.
    
    Returns:
        str: The generated text.
    """
    res = tokenizer.tokenize(prefix)
    pred = ""
    count = 0
    while count < max_len and pred != stop_token:
        pred = word_pred[tuple(res[-n+1:])]
        res.append(pred)
        count += 1
    return tokenizer.convert_tokens_to_string(res)


def plot_next_word_prob(word_scores, word_candidates, context, top=10, save_path=None):
    """Plot the probabilities of the next word for a given context.
    
    Args:
        word_scores: The scores of the next words.
        word_candidates: The next word candidates.
        context: The context for which to plot the probabilities.
        top: The number of top probabilities to plot.
        save_path: The path to save the plot.
    """
    scores = np.array(word_scores[context])
    candidates = word_candidates[context]
    top_indices = np.argsort(scores)[-top:][::-1]
    top_scores = scores[top_indices]
    top_candidates = [candidates[i] for i in top_indices]

    plt.figure(figsize=(10, 6))
    plt.bar(range(top), top_scores, tick_label=top_candidates)
    plt.xticks(rotation=90)
    plt.xlabel('Next words')
    plt.ylabel('Probability')
    plt.title(f'Top {top} next word probabilities for context: {" ".join(context)}')
    if save_path:
        plt.savefig(save_path)
    plt.show()


def run_ngram():
    """Run the n-gram model."""
    train_data, dev_data = load_data()
    splitter = SentenceSplitter(language='en')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    sentence_split_and_tokenize_demo(train_data, splitter, tokenizer)
    word_pred, next_word_scores, next_word_candidates = create_ngrams(train_data[:100000], 3, splitter, tokenizer)

    context1 = ("move", "to")
    context2 = ("the", "news")
    print(f"{'-' * 10} plot the top 10 next word probabilities after {context1} {'-' * 10}")
    plot_next_word_prob(next_word_scores, next_word_candidates, context1, top=10, save_path="ngram_context1.png")
    print(f"{'-' * 10} plot the top 10 next word probabilities after {context2} {'-' * 10}")
    plot_next_word_prob(next_word_scores, next_word_candidates, context2, top=10, save_path="ngram_context2.png")

    prefix1 = "According to the report"
    prefix2 = "The president of the association"
    completion1 = generate_text(word_pred, tokenizer, 3, prefix1, max_len=30)
    completion2 = generate_text(word_pred, tokenizer, 3, prefix2, max_len=30)
    print(f"{'-' * 10} generated text 1 {'-' * 10}")
    print(completion1)
    print(f"{'-' * 10} generated text 2 {'-' * 10}")
    print(completion2)
