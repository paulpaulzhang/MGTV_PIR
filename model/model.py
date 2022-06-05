from torch import nn
import torch


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


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
        self.meanpool = MeanPooling()

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
        cls_output = outputs[0][:, 0, :]

        output, (h_n, c_n) = self.bilstm(torch.tanh(sequence_output))

        pooled_output = self.meanpool(output, attention_mask)

        logits = self.classifier(torch.tanh(pooled_output))

        return (logits, cls_output)
