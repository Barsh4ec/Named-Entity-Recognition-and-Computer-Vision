import torch
from transformers import BertForTokenClassification, BertTokenizerFast


unique_labels = {"B-geo", "I-geo", "O"}
labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
ids_to_labels = {v: k for v, k in enumerate(unique_labels)}
label_all_tokens = False
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")


class BertModel(torch.nn.Module):

    def __init__(self):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output
