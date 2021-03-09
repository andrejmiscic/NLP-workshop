import itertools
import random

import torch
from torch.utils.data import Dataset


class TextDatasetWithEpochs(Dataset):  # based on TextDataset by Huggingface
    def __init__(self, examples, tokenizer, block_size, num_epochs, example_del="<|endoftext|>"):
        super(TextDatasetWithEpochs, self).__init__()
        examples_input_ids = []
        for ex in examples:
            # we add the delimeter to each quote, tokenize it and convert the tokens to indices
            examples_input_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example_del + ex)))

        # for each of the training epochs shuffle the quotes and combined them
        combined_input_ids = []
        for i in range(num_epochs):
            tmp = examples_input_ids.copy()
            random.shuffle(tmp)
            combined_input_ids.extend(list(itertools.chain.from_iterable(tmp)))

        # creating training samples by cutting the combined input into blocks of length block_size
        self.data = []
        for i in range(0, len(combined_input_ids) - block_size + 1, block_size):
            self.data.append(tokenizer.build_inputs_with_special_tokens(combined_input_ids[i: i + block_size]))

    def __getitem__(self, i):
        return torch.tensor(self.data[i], dtype=torch.long)

    def __len__(self):
        return len(self.data)
