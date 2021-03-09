import os

from dataclasses import dataclass

import numpy as np
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup, AdamW


class Trainer:
    def __init__(self, model):
        self.model = model

    def train(self, train_dataset, val_dataset, device, run_config):
        self.model = self.model.to(device)
        # create output folder if it doesn't yet exist
        if not os.path.isdir(run_config.output_dir):
            os.makedirs(run_config.output_dir)

        # train dataloader will serve us the training data in batches
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                      batch_size=run_config.batch_size, collate_fn=run_config.collate_fn)

        # optimizer and scheduler that modifies the learning rate during the training
        optimizer = AdamW(self.model.parameters(), lr=run_config.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=run_config.num_warmup_steps,
                                                    num_training_steps=len(train_dataloader) * run_config.num_epochs)

        print("Training started:")
        print(f"\tNum examples = {len(train_dataset)}")
        print(f"\tNum Epochs = {run_config.num_epochs}")

        global_step = 0  # to save after every save_steps if save_steps is >= 0

        train_iterator = trange(0, int(run_config.num_epochs), desc="Epoch")
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)
            self.model.train()
            epoch_losses = []
            for step, inputs in enumerate(epoch_iterator):
                # move batch to GPU
                if isinstance(inputs, dict):
                    for k, v in inputs.items():
                        inputs[k] = v.to(device)
                else:
                    inputs = inputs.to(device)

                # forward pass - model also outputs a computed loss
                outputs = self.model(**inputs)
                loss = outputs[0]

                epoch_losses.append(loss.item())

                # backward pass - backpropagation
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_iterator.set_description(f"Training loss = {loss.item():.4f}")

                if run_config.save_steps > -1 and global_step > 0 and global_step % run_config.save_steps == 0:
                    output_dir = os.path.join(run_config.output_dir, f"Step_{step}")
                    self.model.save_pretrained(output_dir)
                    test_loss = self.evaluate(self.model, val_dataset, device, run_config)
                    print(f"After step {step + 1}: val loss ={test_loss}")

                global_step += 1

            if run_config.save_each_epoch or epoch == run_config.num_epochs - 1:
                output_dir = os.path.join(run_config.output_dir, f"Epoch_{epoch + 1}")
                self.model.save_pretrained(output_dir)
            test_loss = self.evaluate(self.model, val_dataset, device, run_config)
            print(f"After epoch {epoch + 1}: val loss ={test_loss}")

    def evaluate(self, model, test_dataset, device, run_config):
        test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                     batch_size=run_config.batch_size, collate_fn=run_config.collate_fn)
        self.model.eval()
        losses = []
        for inputs in tqdm(test_dataloader, desc="Evaluating", position=0, leave=True):
            # move batch to GPU
            if isinstance(inputs, dict):
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
            else:
                inputs = inputs.to(device)

            with torch.no_grad():
                loss = model(**inputs)[0]
            losses.append(loss.item())

        return np.mean(losses)


@dataclass
class RunConfig:
    learning_rate: float
    batch_size: int
    num_epochs: int
    num_warmup_steps: int = 1
    save_steps: int = -1
    save_each_epoch: bool = False
    output_dir: str = "/content/model/"
    collate_fn: None = None
