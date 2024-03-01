import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn as nn 
from torch.utils.data import DataLoader
from datasets import load_dataset
import evaluate as evaluate
from transformers import get_scheduler
from transformers import AutoModel, AutoModelForSequenceClassification
import argparse
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm

import time

# Related to BERT
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings

class CustomModelforSequenceClassification(nn.Module):

    def __init__(self, model_name, num_labels=2, type="full"):
        super(CustomModelforSequenceClassification, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.type = type
        self.num_labels = num_labels
        self.prefix = torch.nn.Parameter(torch.randn(prefix_length, self.model.config.hidden_size, requires_grad=True).to('cuda'))
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        
        if self.type == "full":
            # TODO: implement the forward function for the full model
            raise NotImplementedError("You need to implement the forward function for the full model")

            # pass the input_ids and attention_mask into the model to get the output object (you can name it `output`)

            # get the last hidden state from the output object using `.last_hidden_state`

            # take the mean of the last hidden state along the sequence length dimension

            # pass the mean into the self.classifier to get the logits

            # your code ends here

        elif self.type == "head":
            # TODO: implement the forward function for the head-tuned model
            raise NotImplementedError("You need to implement the forward function for the head-tuned model")
            # Hint: it should be the same as the full model

            # your code ends here
        
        elif self.type == 'prefix':

            # TODO: implement the forward function for the prefix-tuned model
            raise NotImplementedError("You need to implement the forward function for the prefix-tuned model")

            # the prefix is at self.prefix, but this is only one prefix, we want to append it to each instance in a batch
            # we make multiple copies of self.prefix here. the number of copies = batch size


            
            # concatentate the input embeddings and our prefix, make sure to put them into our gpu
            # get the input embeddings
            # Hint: you can use self.model.embeddings.word_embeddings to get the input embeddings

            # concatenate the input embeddings and the prefix
            # Hint: check torch.cat for how to concatenate the tensors

            # move the input embeddings to the gpu
            # Hint: use .to(device='cuda') to move the tensor to the gpu
            # name the final tensor as `inputs_embeds`

            
            # modify attention mask
            # we need to add the prefix to the attention mask
            # the mask on the prefix should be 1, with the dimension of (batch_size, prefix_length)
            # name the final attention mask as `attention_mask`



            # pass the input embeddings and the attention mask into the model
            # you can do this by passing a keyword argument "inputs_embeds" to model.forward

            # your code ends here

            # get the last hidden state from the output object, take the mean, and pass it into the classifier
            # Hint: same as the full model and head-tuned model



            # your code ends here

        return {"logits": logits}
    
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
    """
    Evaluate a PyTorch Model
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

        predictions = output['logits']
        predictions = torch.argmax(predictions, dim=1)
        dev_accuracy.add_batch(predictions=predictions, references=batch['labels'])

    # compute and return metrics
    return dev_accuracy.compute()


def train(mymodel, num_epochs, train_dataloader, validation_dataloader, test_dataloder, device, lr, model_name):
    """ Train a PyTorch Module

    :param torch.nn.Module mymodel: the model to be trained
    :param int num_epochs: number of epochs to train for
    :param torch.utils.data.DataLoader train_dataloader: DataLoader containing training examples
    :param torch.utils.data.DataLoader validation_dataloader: DataLoader containing validation examples
    :param torch.device device: the device that we'll be training on
    :param float lr: learning rate
    :param string model_name: the name of the model
    :return None
    """

    # here, we use the AdamW optimizer. Use torch.optim.AdamW
    print(" >>>>>>>>  Initializing optimizer")
    
    weight_decay = 0.01
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in mymodel.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay},
        {'params': [p for n, p in mymodel.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    # need to customize optimizer for prefix-tuning and head tuning

    if mymodel.type == "head":
        # TODO: implement the optimizer for head-tuned model
        raise NotImplementedError("You need to implement the optimizer for head-tuned model")
        # you need to get the parameters of the classifier (head), you can do this by calling mymodel.head.parameters()

        # then you need to pass these parameters to the optimizer
        # name the optimizer as `custom_optimizer`
        # Hints: you can refer to how we do this for the optimizer above

        # your code ends here
    
    elif mymodel.type == "prefix":
        # TODO: implement the optimizer for prefix-tuned model
        raise NotImplementedError("You need to implement the optimizer for prefix-tuned model")
        # you need to get the parameters of the prefix, you can do this by calling mymodel.prefix
        # name the parameters as `prefix_params`

        # you also need to get the parameters of the classifier (head), you can do this by calling mymodel.head.parameters()
        # name the parameters as `classifier_params`

        # your code ends here
        # group the parameters together
        custom_optimizer = torch.optim.AdamW([prefix_params] + list(classifier_params), lr=lr)


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

        epoch_start_time = time.time()

        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        mymodel.train()

        # load metrics
        train_accuracy = evaluate.load('accuracy')

        print(f"Epoch {epoch + 1} training:")

        for index, batch in tqdm(enumerate(train_dataloader)):

            """
            You need to make some changes here to make this function work.
            Specifically, you need to: 
            Extract the input_ids, attention_mask, and labels from the batch; then send them to the device. 
            Then, pass the input_ids and attention_mask to the model to get the logits.
            Then, compute the loss using the logits and the labels.
            Then, depending on model.type, you may want to use different optimizers
            Then, call loss.backward() to compute the gradients.
            Then, call lr_scheduler.step() to update the learning rate.
            Then, call optimizer.step()  to update the model parameters.
            Then, call optimizer.zero_grad() to reset the gradients for the next iteration.
            Then, compute the accuracy using the logits and the labels.
            """

            # TODO: implement the training loop
            raise NotImplementedError("You need to implement this function")

            # get the input_ids, attention_mask, and labels from the batch and put them on the device
            # Hints: similar to the evaluate_model function



            # forward pass
            # name the output as `output`
            # Hints: refer to the evaluate_model function on how to get the predictions (logits)
            # - It's slightly different from the implementation in train of base_classification.py



            # compute the loss using the loss function


            # loss backward

            # your code ends here

            # update the model parameters depending on the model type
            if mymodel.type == "full" or mymodel.type == "auto":
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            else:
                custom_optimizer.step()
                lr_scheduler.step()
                custom_optimizer.zero_grad()

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
        
        test_accuracy = evaluate_model(mymodel, test_dataloader, device)
        print(f" - Average test metrics: accuracy={test_accuracy}")

        epoch_end_time = time.time()
        print(f"Epoch {epoch + 1} took {epoch_end_time - epoch_start_time} seconds")

    plot(train_acc_list, dev_acc_list, name=model_name, finetune_method=mymodel.type)

def plot(train_list, valid_list, name, finetune_method):
    
    plt.figure()
    plt.plot(train_list, label='Train')
    plt.plot(valid_list, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()
    plt.savefig(f'{name}_{finetune_method}.png')


def pre_process(model_name, batch_size, device, small_subset, type='auto'):
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
    mytokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

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
    
    if type == "auto":
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    else:
        pretrained_model = CustomModelforSequenceClassification(model_name, num_labels=2, type=type)


    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)
    return pretrained_model, train_dataloader, validation_dataloader, test_dataloader


# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_subset", action='store_true')
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--type", type=str, default="auto", choices=["auto", "full", "head", "prefix"], help="type of tuning to perform on the model")
    parser.add_argument("--prefix_length", type=int, default=128)
    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    assert type(args.small_subset) == bool, "small_subset must be a boolean"
    global prefix_length
    prefix_length = args.prefix_length
    #load the data and models
    pretrained_model, train_dataloader, validation_dataloader, test_dataloader = pre_process(args.model,
                                                                                             args.batch_size,
                                                                                             args.device,
                                                                                             args.small_subset,
                                                                                             args.type)
    print(" >>>>>>>>  Starting training ... ")
    train(pretrained_model, args.num_epochs, train_dataloader, validation_dataloader, test_dataloader, args.device, args.lr, args.model)
    
    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory()

    val_accuracy = evaluate_model(pretrained_model, validation_dataloader, args.device)
    print(f" - Average DEV metrics: accuracy={val_accuracy}")

    test_accuracy = evaluate_model(pretrained_model, test_dataloader, args.device)
    print(f" - Average TEST metrics: accuracy={test_accuracy}")
