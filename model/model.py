from torch import nn
import torch


class BertForSequenceClassification(nn.Module):
    def __init__(self, config, bert):
        super().__init__()

        self.num_labels = config.num_labels
        self.config = config

        self.bert = bert

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        **kargs
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)

        return (logits,)


class BertBiLSTMForSequenceClassification(nn.Module):
    def __init__(self, config, bert):
        super().__init__()

        self.num_labels = config.num_labels
        self.config = config

        self.bert = bert

        self.bilstm = nn.LSTM(input_size=config.hidden_size,
                              hidden_size=config.hidden_size // 2,
                              bidirectional=True,
                              dropout=0.1,
                              batch_first=True)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        **kargs
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        sequence_output = outputs[0]
        cls_output = outputs[1]

        output, (h_n, c_n) = self.bilstm(sequence_output)

        pooled_output = output[:, 0]

        logits = self.classifier(pooled_output)

        return (logits, cls_output)
