from easydict import EasyDict
from mlp_lm import run_mlp_lm, load_data_mlp_lm, sample_from_mlp_lm, visualize_epochs
from tokenization import test_one_step_bpe, test_bpe, bpe_on_wikitext

def single_run_mlp_lm(train_d, dev_d):
    train_config = EasyDict({
        # model configuration
        'embed_dim': 128,  # the dimension of the word embeddings
        'hidden_dim': 512,  # the dimension of the hidden layer
        'num_blocks': 2,  # the number of transformer blocks
        'dropout_p': 0.2,  # the probability of dropout
        'local_window_size': 6,  # the size of the local window
        # training configuration
        'batch_size': 4096,  # batch size
        'lr': 2e-6,  # learning rate
        'decay': 1.0,
        'num_epochs': 5,  # the total number of times all the training data is iterated over
        'save_path': 'model.pth',  # path where to save the model
    })

    epoch_train_losses, epoch_train_ppls, epoch_dev_losses, epoch_dev_ppls = run_mlp_lm(train_config, train_d, dev_d)
    visualize_epochs(epoch_train_losses, epoch_dev_losses, "Loss", "mlp_lm_loss.png")
    visualize_epochs(epoch_train_ppls, epoch_dev_ppls, "Perplexity", "mlp_lm_ppl.png")


def sample_from_trained_mlp_lm(dev_d):
    pretrained_config = EasyDict({
        # model configuration
        'embed_dim': 256,  # the dimension of the word embeddings
        'hidden_dim': 1048,  # the dimension of the hidden layer
        'num_blocks': 4,  # the number of transformer blocks
        'dropout_p': 0.2,  # the probability of dropout
        'local_window_size': 6,  # the size of the local window
        'save_path': 'pretrained_fixed_window_lm.dat',  # path where to save the model
        # evaluation configuration
        'batch_size': 4096,  # batch size
    })
    sample_from_mlp_lm(pretrained_config, dev_d)


if __name__ == '__main__':
    # Test the one-step BPE algorithm
    # uncomment the following line to run
    # test_one_step_bpe()

    # Test the BPE algorithm
    # uncomment the following line to run
    # test_bpe()

    # Run BPE on the wikitext dataset
    # uncomment the following line to run
    # bpe_on_wikitext()

    # load raw data for lm
    # uncomment the following line to run
    # train_data, dev_data = load_data_mlp_lm()

    # Run a single training run
    # uncomment the following line to run
    # single_run_mlp_lm(train_data, dev_data)

    # Sample from the pretrained model
    # uncomment the following line to run
    # sample_from_trained_mlp_lm(dev_data)

