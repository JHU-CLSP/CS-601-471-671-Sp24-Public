import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import evaluate as evaluate
from transformers import get_scheduler
from transformers import AutoModelForSequenceClassification
import argparse
import subprocess
import matplotlib.pyplot as plt

def print_gpu_memory():
    """
    Print the amount of GPU memory used by the current process
    This is useful for debugging memory issues on the GPU
    """
    # check if gpu is available
    if torch.cuda.is_available():
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

        p = subprocess.check_output('nvidia-smi')
        print(p.decode("utf-8"))


class BoolQADataset(torch.utils.data.Dataset):
    """
    Dataset for the dataset of BoolQ questions and answers
    """

    def __init__(self, passages, questions, answers, tokenizer, max_len):
        self.passages = passages
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, index):
        """
        This function is called by the DataLoader to get an instance of the data
        :param index:
        :return:
        """

        passage = str(self.passages[index])
        question = self.questions[index]
        answer = self.answers[index]

        # this is input encoding for your model. Note, question comes first since we are doing question answering
        # and we don't wnt it to be truncated if the passage is too long
        input_encoding = question + " [SEP] " + passage

        # encode_plus will encode the input and return a dictionary of tensors
        encoded_review = self.tokenizer.encode_plus(
            input_encoding,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        return {
            'input_ids': encoded_review['input_ids'][0],  # we only have one example in the batch
            'attention_mask': encoded_review['attention_mask'][0],
            # attention mask tells the model where tokens are padding
            'labels': torch.tensor(answer, dtype=torch.long)  # labels are the answers (yes/no)
        }


def evaluate_model(model, dataloader, device):
    """ Evaluate a PyTorch Model
    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on
    :return accuracy
    """

    # load metrics
    dev_accuracy = evaluate.load('accuracy')

    # turn model into evaluation mode
    model.eval()

    # iterate over the dataloader
    for batch in dataloader:
        # TODO: implement the evaluation function
        raise NotImplementedError("You need to implement the evaluation function")
        # get the input_ids, attention_mask from the batch and put them on the device
        # Hints:
        # - see the getitem function in the BoolQADataset class for how to access the input_ids and attention_mask
        # - use to() to move the tensors to the device


        # forward pass
        # name the output as `output`

        # your code ends here

        predictions = output.logits
        predictions = torch.argmax(predictions, dim=1)
        dev_accuracy.add_batch(predictions=predictions, references=batch['labels'])

    # compute and return metrics
    return dev_accuracy.compute()


def train(mymodel, num_epochs, train_dataloader, validation_dataloader, test_dataloder, device, lr, small_subset=False):
    """ Train a PyTorch Module

    :param torch.nn.Module mymodel: the model to be trained
    :param int num_epochs: number of epochs to train for
    :param torch.utils.data.DataLoader train_dataloader: DataLoader containing training examples
    :param torch.utils.data.DataLoader validation_dataloader: DataLoader containing validation examples
    :param torch.device device: the device that we'll be training on
    :param float lr: learning rate
    :return None
    """

    # here, we use the AdamW optimizer. Use torch.optim.Adam.
    # instantiate it on the untrained model parameters with a learning rate of 5e-5
    print(" >>>>>>>>  Initializing optimizer")
    optimizer = torch.optim.AdamW(mymodel.parameters(), lr=lr)

    # now, we set up the learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    loss = torch.nn.CrossEntropyLoss()
    
    epoch_list = []
    train_acc_list = []
    dev_acc_list = []

    for epoch in range(num_epochs):

        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        mymodel.train()

        # load metrics
        train_accuracy = evaluate.load('accuracy')

        print(f"Epoch {epoch + 1} training:")

        for i, batch in tqdm(enumerate(train_dataloader)):

            """
            You need to make some changes here to make this function work.
            Specifically, you need to: 
            Extract the input_ids, attention_mask, and labels from the batch; then send them to the device. 
            Then, pass the input_ids and attention_mask to the model to get the logits.
            Then, compute the loss using the logits and the labels.
            Then, call loss.backward() to compute the gradients.
            Then, call optimizer.step()  to update the model parameters.
            Then, call lr_scheduler.step() to update the learning rate.
            Then, call optimizer.zero_grad() to reset the gradients for the next iteration.
            """

            # TODO: implement the training loop
            raise NotImplementedError("You need to implement this function")

            # get the input_ids, attention_mask, and labels from the batch and put them on the device
            # Hints: similar to the evaluate_model function


            # forward pass
            # name the output as `output`
            # Hints: refer to the evaluate_model function on how to get the predictions (logits)


            # compute the loss using the loss function


            # loss backward


            # update the model parameters with optimizer and lr_scheduler step


            # zero the gradients

            # your code ends here

            predictions = torch.argmax(predictions, dim=1)

            # update metrics
            train_accuracy.add_batch(predictions=predictions, references=batch['labels'])
            
        # print evaluation metrics
        print(f" ===> Epoch {epoch + 1}")
        train_acc = train_accuracy.compute()
        print(f" - Average training metrics: accuracy={train_acc}")
        train_acc_list.append(train_acc['accuracy'])

        # normally, validation would be more useful when training for many epochs
        val_accuracy = evaluate_model(mymodel, validation_dataloader, device)
        print(f" - Average validation metrics: accuracy={val_accuracy}")
        dev_acc_list.append(val_accuracy['accuracy'])
        
        epoch_list.append(epoch)
        
    # generate plots here
    plt.clf()
    plt.plot(epoch_list, train_acc_list, 'b', label='train')
    if not small_subset:
        plt.plot(epoch_list, dev_acc_list, 'g', label='valid')
    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    save_path = "overfit.png" if small_subset else "base_full.png"
    plt.savefig(save_path)

def pre_process(model_name, batch_size, device, small_subset):
    # download dataset
    print("Loading the dataset ...")
    dataset = load_dataset("boolq")
    dataset = dataset.shuffle()  # shuffle the data

    print("Slicing the data...")
    if small_subset:
        # use this tiny subset for debugging the implementation
        dataset_train_subset = dataset['train'][:10]
        dataset_dev_subset = dataset['train'][:10]
        dataset_test_subset = dataset['train'][:10]
    else:
        # since the dataset does not come with any validation data,
        # split the training data into "train" and "dev"
        dataset_train_subset = dataset['train'][:8000]
        dataset_dev_subset = dataset['validation']
        dataset_test_subset = dataset['train'][8000:]

    print("Size of the loaded dataset:")
    print(f" - train: {len(dataset_train_subset['passage'])}")
    print(f" - dev: {len(dataset_dev_subset['passage'])}")
    print(f" - test: {len(dataset_test_subset['passage'])}")

    # maximum length of the input; any input longer than this will be truncated
    # we had to do some pre-processing on the data to figure what is the length of most instances in the dataset
    max_len = 128

    print("Loading the tokenizer...")
    mytokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loding the data into DS...")
    train_dataset = BoolQADataset(
        passages=list(dataset_train_subset['passage']),
        questions=list(dataset_train_subset['question']),
        answers=list(dataset_train_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )
    validation_dataset = BoolQADataset(
        passages=list(dataset_dev_subset['passage']),
        questions=list(dataset_dev_subset['question']),
        answers=list(dataset_dev_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )
    test_dataset = BoolQADataset(
        passages=list(dataset_test_subset['passage']),
        questions=list(dataset_test_subset['question']),
        answers=list(dataset_test_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )

    print(" >>>>>>>> Initializing the data loaders ... ")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # from Hugging Face (transformers), read their documentation to do this.
    print("Loading the model ...")
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)
    return pretrained_model, train_dataloader, validation_dataloader, test_dataloader


# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_subset", action='store_true',
                        help="When set true, only run training on a small subset of the data, used for 3.1.1")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")

    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    assert type(args.small_subset) == bool, "small_subset must be a boolean"

    # load the data and models
    pretrained_model, train_dataloader, validation_dataloader, test_dataloader = pre_process(args.model,
                                                                                             args.batch_size,
                                                                                             args.device,
                                                                                             args.small_subset)
    print(" >>>>>>>>  Starting training ... ")
    train(pretrained_model, args.num_epochs, train_dataloader, validation_dataloader, test_dataloader, args.device, args.lr, args.small_subset)
    
    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory()

    val_accuracy = evaluate_model(pretrained_model, validation_dataloader, args.device)
    print(f" - Average DEV metrics: accuracy={val_accuracy}")

    test_accuracy = evaluate_model(pretrained_model, test_dataloader, args.device)
    print(f" - Average TEST metrics: accuracy={test_accuracy}")
