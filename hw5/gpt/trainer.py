import time
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from gpt.utils import CfgNode as CN
from matplotlib import pyplot as plt

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'cuda'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataloader, dev_dataloader):
        self.config = config
        self.model = model
        self.optimizer = None
        # self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.iter_train_loss = 0.0
        self.iter_train_ppl = 0.0
        self.all_iter_train_loss = []
        self.all_iter_train_ppl = []
        self.valid_loss = 0.0
        self.valid_ppl = 0.0
        self.all_iter_valid_loss = []
        self.all_iter_valid_ppl = []
        self.best_valid_ppl = float('inf')

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def evaluation(self):
        model, config = self.model, self.config
        model.eval()
        valid_loss = []
        valid_ppls = []
        with torch.no_grad():
            for batch in self.dev_dataloader:
                batch = [t.to(self.device) for t in batch]
                input_ids, labels, masks = batch
                logits, loss = model(input_ids, labels, masks)
                valid_loss.append(loss.item())
                valid_ppls.append(torch.exp(loss).item())
        self.valid_loss = sum(valid_loss) / len(valid_loss)
        self.valid_ppl = sum(valid_ppls) / len(valid_ppls)
        self.trigger_callbacks('on_validation_end')
        self.all_iter_valid_loss.append(self.valid_loss)
        self.all_iter_valid_ppl.append(self.valid_ppl)
        model.train()

    def plot(self):
        plt.clf()
        plt.plot(list(range(0, len(self.all_iter_train_loss))), self.all_iter_train_loss, label='train loss')
        plt.plot(list(range(0, len(self.all_iter_valid_loss) * 1000, 1000)), self.all_iter_valid_loss, label='valid loss')
        plt.xticks(np.arange(0, len(self.all_iter_valid_loss) * 1000, 10000).astype(np.int32))
        plt.xlabel('Steps')
        plt.ylabel(f'Loss')
        plt.legend()
        plt.savefig("loss.png")

        plt.clf()
        plt.plot(list(range(0, len(self.all_iter_train_ppl))), self.all_iter_train_ppl, label='train ppl')
        plt.plot(list(range(0, len(self.all_iter_valid_ppl) * 1000, 1000)), self.all_iter_valid_ppl, label='valid ppl')
        plt.xticks(np.arange(0, len(self.all_iter_valid_ppl) * 1000, 10000).astype(np.int32))
        plt.xlabel('Steps')
        plt.ylabel(f'Perplexity')
        plt.yscale('log')
        plt.legend()
        plt.savefig("ppl.png")

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(self.train_dataloader)
        # run evaluation before training
        self.evaluation()
        start_time = time.time()
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            input_ids, labels, masks = batch

            # forward the model
            # logits, self.loss = model(input_ids, labels)
            logits, self.iter_train_loss = model(input_ids, labels, masks)
            self.iter_train_ppl = torch.exp(self.iter_train_loss)
            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.iter_train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # evaluate the model
            if self.iter_num % 1000 == 0:
                self.evaluation()

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

        print(f"{'-'*10} Trainer Time Elapsed: {time.time() - start_time:.2f}s {'-'*10}\n")
