import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, SequentialSampler
from tqdm import tqdm


class TextClassificationDataset(Dataset):
    def __init__(self, inputs, labels, tokenizer, max_len):
        super(TextClassificationDataset, self).__init__()

        # encodes the inputs to input_ids and attention_mask
        encoded_inputs = tokenizer(inputs, max_length=max_len, padding="max_length", truncation=True)
        self.data = list(zip(encoded_inputs["input_ids"], encoded_inputs["attention_mask"], labels))

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


def collate_batch_to_tensors(inputs):
    batch = {"input_ids": torch.tensor([dat[0] for dat in inputs], dtype=torch.long),
             "attention_mask": torch.tensor([dat[1] for dat in inputs], dtype=torch.long),
             "labels": torch.tensor([dat[2] for dat in inputs], dtype=torch.long)}
    return batch


def seq_cls_evaluate(model, test_dataset, device, batch_size=32):
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                 batch_size=batch_size, collate_fn=collate_batch_to_tensors)
    model.eval()
    ce_losses, acc_losses = [], []
    with torch.no_grad():
        for inputs in tqdm(test_dataloader, desc="Evaluating", position=0, leave=True):
            # move batch to GPU
            if isinstance(inputs, dict):
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
            else:
                inputs = inputs.to(device)

            loss, logits_y = model(**inputs)
            ce_losses.append(loss.item())
            pred_y = np.argmax(nn.functional.softmax(logits_y, dim=1).squeeze().cpu().numpy(), axis=1)  # beautiful
            true_y = inputs["labels"].cpu().numpy()
            acc_losses.append(np.mean(pred_y == true_y))

    return np.mean(ce_losses), np.mean(acc_losses)
