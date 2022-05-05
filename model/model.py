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

        return logits


class BertEnsambleForSequenceClassification(nn.Module):
    def __init__(self, config, bert):
        super().__init__()

        self.num_labels = config.num_labels
        self.config = config

        self.bert = bert

        self.bilstm1 = nn.LSTM(input_size=config.hidden_size,
                              hidden_size=config.hidden_size // 2,
                              bidirectional=True,
                              dropout=0.1,
                              batch_first=True)
        self.bilstm2 = nn.LSTM(input_size=config.hidden_size,
                        hidden_size=config.hidden_size // 2,
                        bidirectional=True,
                        dropout=0.1,
                        batch_first=True)
        self.bilstm3 = nn.LSTM(input_size=config.hidden_size,
                        hidden_size=config.hidden_size // 2,
                        bidirectional=True,
                        dropout=0.1,
                        batch_first=True)
        self.bilstm4 = nn.LSTM(input_size=config.hidden_size,
                              hidden_size=config.hidden_size // 2,
                              bidirectional=True,
                              dropout=0.1,
                              batch_first=True)

        self.classifier1 = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier2 = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier3 = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier4 = nn.Linear(config.hidden_size, config.num_labels)

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
            #   output_attentions=False,
            # output_hidden_states=True,
        )

        all_hidden_states = outputs[2]

        output1, (h_n, c_n) = self.bilstm(all_hidden_states[-1])
        output2, (h_n, c_n) = self.bilstm(all_hidden_states[-2])
        output3, (h_n, c_n) = self.bilstm(all_hidden_states[-3])
        output4, (h_n, c_n) = self.bilstm(all_hidden_states[-4])

        logits1 = self.classifier1(output1[:, 0])
        logits2 = self.classifier2(output2[:, 0])
        logits3 = self.classifier3(output3[:, 0])
        logits4 = self.classifier4(output4[:, 0])

        logits = logits1 + logits2 + logits3 + logits4
        return logits


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

        output, (h_n, c_n) = self.bilstm(sequence_output)

        pooled_output = output[:, 0]

        logits = self.classifier(pooled_output)

        return logits


