from data import load_data, create_datasets, create_dataloader, VOCAB_SIZE
import torch
from gpt.model import GPT
from gpt.trainer import Trainer
from generate import generate


def batch_end_callback(trainer):
    trainer.all_iter_train_loss.append(trainer.iter_train_loss.item())
    trainer.all_iter_train_ppl.append(trainer.iter_train_ppl.item())
    if trainer.iter_num % 100 == 0:
        print(
            f"iter_dt {trainer.iter_dt * 100:.2f}s; iter {trainer.iter_num}: "
            f"train loss {trainer.iter_train_loss.item():.5f} train ppl {trainer.iter_train_ppl.item():.3f}")


def evaluation_callback(trainer):
    print(f"iter {trainer.iter_num}: validation loss {trainer.valid_loss:.5f} validation ppl {trainer.valid_ppl:.3f}")
    if trainer.valid_ppl < trainer.best_valid_ppl:
        print(f"best model so far, saving...")
        torch.save(trainer.model.state_dict(), 'model.pth')
        trainer.best_valid_ppl = trainer.valid_ppl


def sample_from_trained_model(trained_model):
    trained_model.load_state_dict(torch.load('model.pth'))
    trained_model.to('cuda')
    trained_model.eval()

    generate(trained_model, prompt=' The best perks of living on the east', num_samples=1, steps=30)
    generate(trained_model, prompt=' It is a truth universally acknowledged', num_samples=1, steps=30)
    generate(trained_model, prompt=' The best way to learn', num_samples=1, steps=30)
    generate(trained_model, prompt=' According to the latest research', num_samples=1, steps=30)


def run(train_dataset, dev_dataset, max_iter=1, device='cpu', plot=True, sample=True):
    # create dataloaders
    train_dataloader = create_dataloader(train_dataset, batch_size=256, shuffle=True)
    dev_dataloader = create_dataloader(dev_dataset, batch_size=256, shuffle=False)

    # create model
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = VOCAB_SIZE
    model_config.block_size = 32
    model = GPT(model_config)

    # create trainer
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4  # the model we're using is so small that we can go a bit faster
    train_config.max_iters = max_iter
    train_config.num_workers = 0
    train_config.device = device
    trainer = Trainer(train_config, model, train_dataloader, dev_dataloader)
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.set_callback('on_validation_end', evaluation_callback)

    trainer.run()
    if plot:
        trainer.plot()
    if sample:
        sample_from_trained_model(model)

def cpu_gpu_comparison(train_dataset, dev_dataset, max_iter=100):
    run(train_dataset, dev_dataset, max_iter=max_iter, device='cpu', plot=False, sample=False)
    run(train_dataset, dev_dataset, max_iter=max_iter, device='cuda', plot=False, sample=False)


def gpu_full_run(train_dataset, dev_dataset, max_iter=20000):
    run(train_dataset, dev_dataset, max_iter=max_iter, device='cuda', plot=True, sample=True)


if __name__ == '__main__':
    # load raw data for lm
    train_data, dev_data = load_data()

    # create datasets
    train_d, dev_d = create_datasets(train_data, dev_data, block_size=32)

    # TODO: run cpu & gpu comparison
    # uncomment the following line to run
    # cpu_gpu_comparison(train_d, dev_d, max_iter=50)

    # TODO: run the full training on gpu
    # uncomment the following line to run
    # gpu_full_run(train_d, dev_d, max_iter=20000)

