import numpy as np
import torch
import torch.nn as nn

from seqeval.metrics import f1_score
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from tqdm import tqdm


# this is used for NER and POS classification
class TokenClassificationDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len, sep_token="[SEP]", cls_token="[PAD]", pad_token="[PAD]"):

        # special tokens in BERT: sep separates two sentences, cls contains representation for
        # sequence classification, pad is used to extend inputs to the same length
        self.sep_token, self.cls_token, self.pad_token = sep_token, cls_token, pad_token
        # we use ignore_label_id to ignore padding tokens when computing loss
        self.ignore_label_id = nn.CrossEntropyLoss().ignore_index
        self.max_seq_len = max_seq_len

        # reads sentences/labels from dataset file
        examples, self.class_list = self.read_examples_labels_from_file(data_path)
        self.class_list.sort()

        self.label2id = {label: i for i, label in enumerate(self.class_list)}
        self.id2label = {i: label for i, label in enumerate(self.class_list)}

        self.inputs = self.convert_examples_to_inputs(examples, tokenizer)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i]

    def read_examples_labels_from_file(self, data_path):
        examples = []
        unique_labels = set()
        with open(data_path, encoding="utf-8") as f:
            words, labels = [], []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append({"words": words, "labels": labels})
                        words, labels = [], []
                else:
                    splits = line.strip().split(" ")
                    words.append(splits[0])
                    labels.append(splits[-1])
                    unique_labels.add(splits[-1])
            if words:
                examples.append({"words": words, "labels": labels})
        return examples, list(unique_labels)

    def convert_examples_to_inputs(self, examples, tokenizer):
        inputs = []
        for example in examples:
            inputs.append(
                self.convert_example_to_inputs(
                    tokenizer, example["words"], example["labels"], self.label2id, None, self.max_seq_len,
                    self.sep_token, self.cls_token, self.pad_token, self.ignore_label_id
                )
            )

        return inputs

    @staticmethod
    def convert_example_to_inputs(tokenizer, words, labels=None, label2id=None, class_list=None,
                                  max_seq_len=512, sep_token="[SEP]", cls_token="[PAD]", pad_token="[PAD]",
                                  ignore_index=nn.CrossEntropyLoss().ignore_index):

        assert label2id is not None or class_list is not None, \
            "You have to provide either a list of classes or a mapping from classes to id"

        if label2id is None:
            label2id = {label: i for i, label in enumerate(sorted(class_list))}

        if labels is None:
            labels = ["O"] * len(words)  # some fake labels so this ugly pipeline doesn't break

        tokens, label_ids = [], []
        for i, (word, label) in enumerate(zip(words, labels)):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                label_ids.extend([label2id[label]] + [ignore_index] * (len(word_tokens) - 1))

        # truncate sentence if it's too long:
        if len(tokens) > max_seq_len - 2:  # 2, because we need to add CLS and SEP tokens
            tokens = tokens[:(max_seq_len - 2)]
            label_ids = label_ids[:(max_seq_len - 2)]

        # adding the separator to the end of the sentence
        tokens += [sep_token]
        label_ids += [ignore_index]

        # adding the classifier token to the start of the sentence
        tokens = [cls_token] + tokens
        label_ids = [ignore_index] + label_ids

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(tokens)

        # padding
        padding_length = max_seq_len - len(tokens)
        tokens += [pad_token] * padding_length
        label_ids += [ignore_index] * padding_length
        input_mask += [0] * padding_length

        # convert tokens to input ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        return {"input_ids": input_ids, "labels": label_ids, "attention_mask": input_mask}


# used to combine inputs into a batch
def collate_dict_batch_to_tensors(inputs):
    batch = {}
    for k in inputs[0].keys():
        batch[k] = torch.tensor([dat[k] for dat in inputs], dtype=torch.long)

    return batch


def align_predictions_and_labels(predictions, labels, id2label):
    label_list = [[] for _ in range(labels.shape[0])]
    preds_list = [[] for _ in range(labels.shape[0])]

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] >= 0:  # otherwise, we ignore it
                label_list[i].append(id2label[labels[i][j]])
                preds_list[i].append(id2label[predictions[i][j]])
    return preds_list, label_list


def token_cls_evaluate(model, test_dataset, device, id2label, batch_size=32):
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                 batch_size=batch_size, collate_fn=collate_dict_batch_to_tensors)
    model.eval()
    losses = []
    predictions, gt_labels = None, None
    for inputs in tqdm(test_dataloader, desc="Evaluating", position=0, leave=True):
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        with torch.no_grad():
            loss, logits = model(**inputs)
        losses.append(loss.item())
        if gt_labels is None:
            gt_labels = inputs["labels"].detach().cpu().numpy()
            predictions = logits.detach().cpu().numpy()
        else:
            gt_labels = np.append(gt_labels, inputs["labels"].detach().cpu().numpy(), axis=0)
            predictions = np.append(predictions, logits.detach().cpu().numpy(), axis=0)

    predictions = np.argmax(predictions, axis=2)
    predictions, labels = align_predictions_and_labels(predictions, gt_labels, id2label)

    f1 = f1_score(labels, predictions)

    return np.mean(losses), f1
