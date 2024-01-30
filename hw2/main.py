import gensim.downloader
from easydict import EasyDict
from ngram_lm import run_ngram
from gradient_descent import load_data, run_grad_descent, visualize_epochs
from typing import List, Tuple, Dict, Union
EMBEDDING_TYPES = ["glove-twitter-50", "glove-twitter-100", "glove-twitter-200", "word2vec-google-news-300"]

def single_grad_descent(dev_d: Dict[str, List[Union[str, int]]],
                        train_d: Dict[str, List[Union[str, int]]],
                        test_d: Dict[str, List[Union[str, int]]]):
    # TODO: once you have completed the gradient_descent.py, you can run this function to train and evaluate your model, and visualize the training process with a plot
    train_config = EasyDict({
        'batch_size': 64,  # we use batching for
        'lr': 0.05,  # learning rate
        'num_epochs': 50,  # the total number of times all the training data is iterated over
        'save_path': 'model.pth',  # path where to save the model
        'embeddings': EMBEDDING_TYPES[0],
        'num_classes': 2,
    })

    epoch_train_losses, _, epoch_dev_loss, epoch_dev_accs, _, _ = run_grad_descent(train_config, dev_d, train_d, test_d)
    visualize_epochs(epoch_train_losses, epoch_dev_loss, "gradient_descent_loss.png")


if __name__ == '__main__':
    # run ngram
    # uncomment the following line to run
    run_ngram()

    # load raw data
    # uncomment the following line to run
    dev_data, train_data, test_data = load_data()

    # Run a single training run
    # uncomment the following line to run
    single_grad_descent(dev_data, train_data, test_data)

