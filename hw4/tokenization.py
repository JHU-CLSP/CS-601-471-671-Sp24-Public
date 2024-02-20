import re
import pprint
from tqdm import tqdm
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from datasets import load_dataset


def full_word_tokenization_demo():
    print(f"{'-' * 10} Full Word Tokenization {'-' * 10}")
    text = "There is an 80% chance of rainfall today. We are pretty sure it is going to rain."
    words = text.split(" ")  # split the text on spaces
    tokens = {v: k for k, v in enumerate(words)}  # generate a word to index mapping
    print(tokens)


def get_word_freq(words: List[str]) -> Dict[str, int]:
    # split and whitespace-join each word
    # append the special token `</w>` to each word
    # count the frequency of each word
    word_freq_dict = {}  # a hashmap to maintain word frequencies
    for word in words:
        key = ' '.join(word) + ' </w>'  # append the special token to each word

        if key not in word_freq_dict:
            word_freq_dict[key] = 0
        word_freq_dict[key] += 1

    return word_freq_dict


# counting frequency of character pairs
def get_pairs(word_freq_dict):
    pairs = defaultdict(int)
    for word, freq in word_freq_dict.items():
        # TODO: split the words into tokens based on their white space
        #  for each neighboring token pair `tokens[i], tokens[i+1]`, add their frequency to `pairs`
        # note: in the beginning, tokens == characters; however, later they will grow bigger than characters
        # pairs is a dictionary with tuple of token pairs as keys and their frequency as values
        raise NotImplementedError
        # your code ends here
    return pairs


def get_most_frequent_pair(token_pairs: Dict[Tuple[str, str], int]) -> Tuple[str, str]:
    # get the most frequent pair of tokens from the token pairs
    best_pair = max(token_pairs, key=token_pairs.get)
    return best_pair


def merge_byte_pairs(best_pair: Tuple[str, str], word_freq_dict: Dict[str, int]) -> Dict[str, int]:
    # merge the most frequent pair of tokens in the word frequency dictionary
    # iterate through each word in the word frequency dictionary
    # replace the most frequent pair of tokens (bigram) with a single merged token
    merged_dict = {}
    bigram = re.escape(' '.join(best_pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in word_freq_dict:
        w_out = p.sub(''.join(best_pair), word)
        merged_dict[w_out] = word_freq_dict[word]
    return merged_dict


def get_subword_tokens(word_freq_dict):
    # get the subword tokens from the (merged) word frequency dictionary
    subwords = set()
    for word, freq in word_freq_dict.items():
        tokens = word.split()
        for token in tokens:
            subwords.add(token)
    return subwords


def test_one_step_bpe():
    print(f"{'-' * 10} Test One Step BPE {'-' * 10}")
    print(f"{'-' * 10} Preprocess and Count Word Frequency {'-' * 10}")
    words = ["Hopkins", "Hopkins", "Hopkins", "JHU", "JHU", "Hops", "Hops"]
    word_freq_dict = get_word_freq(words)
    print(f"Word frequency: {word_freq_dict}")

    print(f"{'-' * 10} Counting Frequency of Character Pairs {'-' * 10}")
    token_pairs = get_pairs(word_freq_dict)
    print(f"Token pairs and their frequency: {token_pairs}")
    assert token_pairs[('H', 'o')] == 5
    assert token_pairs[('k', 'i')] == 3

    print(f"{'-' * 10} Get Most Frequent Pair {'-' * 10}")
    best_pair = get_most_frequent_pair(token_pairs)
    print(f"Most frequent pair: {best_pair}")

    print(f"{'-' * 10} Merge Most Frequent Pair {'-' * 10}")
    word_freq_dict_merged = merge_byte_pairs(best_pair, word_freq_dict)
    print(f"Before merge: {word_freq_dict}")
    print(f"After merge: {word_freq_dict_merged}")
    assert str(word_freq_dict) == "{'H o p k i n s </w>': 3, 'J H U </w>': 2, 'H o p s </w>': 2}"
    assert str(word_freq_dict_merged) == "{'Ho p k i n s </w>': 3, 'J H U </w>': 2, 'Ho p s </w>': 2}"

    print(f"{'-' * 10} Get Subword Tokens {'-' * 10}")
    print(f"Tokens before merge: {get_subword_tokens(word_freq_dict)}")
    print(f"Tokens after merge: {get_subword_tokens(word_freq_dict_merged)}")
    print(f"{'-' * 10}\n")


def formatted_print_subwords(subwords: set, chunk_size=5):
    # print the subwords in a formatted way for better visualization
    for idx, subword in enumerate(list(subwords)):
        if idx % chunk_size == 0 and idx != 0:
            el = '\n'
        elif idx == len(subwords) - 1:
            el = ''
        else:
            el = ', '
        print(subword, end=el)
    print()


def exract_bpe_subwords(text, steps):
    # split the text on spaces to get the list of words
    words = text.split(" ")

    # get the frequency of each word
    word_freq_dict = get_word_freq(words)

    subword_tokens = set()

    # BPE iterative extraction
    for i in tqdm(range(steps)):

        # TODO implement the one step of BPE algorithm
        # hint:
        # - you can use the functions you implemented and provided above
        # - should not be more than 3 lines
        raise NotImplementedError
        # extract token pairs and their frequency


        # find the most frequent token pair


        # merge the token pair with highest frequency

        # your code ends here

        # extract the subwords for visualizing them
        subword_tokens = get_subword_tokens(word_freq_dict)

        print_interval = int(steps / 5)  # show 5 prints
        if i % print_interval == 0:
            print(f"Iteration {i}: ")
            print(f" -> Subword tokens: ")
            formatted_print_subwords(subword_tokens, chunk_size=20)
            print(f" -> Number of subword tokens: {len(subword_tokens)}")
            print(f"{'-' * 10}")

    return subword_tokens


def test_bpe():
    print(f"{'-' * 10} Test BPE Tokenization {'-' * 10}")
    subwords = exract_bpe_subwords("Hopkins Hopkins Hopkins JHU JHU Hops Hops", 10)
    print(f"Final subword tokens: {subwords}")
    print(subwords)
    assert len(subwords.difference({'Hopkins</w>', 's</w>', 'Hop', 'JHU</w>'})) == 0
    print(f"{'-' * 10}\n")


def load_bpe_data():
    # load the dataset
    return load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="train")


def bpe_on_wikitext():
    print(f"{'-' * 10} BPE on a subset of Wikitext {'-' * 10}")
    train_dataset = load_bpe_data()

    text = " ".join(train_dataset['text'][-5000:])
    print(f"Text of the subset (last 5000 examples): {text}")
    print(text)

    subwords = exract_bpe_subwords(text, 2000)
    print(f"Final Subword tokens")
    formatted_print_subwords(subwords, chunk_size=20)
