from torch import nn
import torch


class BertForSequenceClassification(nn.Module):
    def __init__(self, config, bert):
        super().__init__()

        self.num_labels = config.num_labels
        self.config = config

        self.bert = bert
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
         #   output_hidden_states=True,
        )

        # pooled_output = outputs[1]
        # logits = self.classifier(pooled_output)

        all_hidden_states = outputs[2]

        pooled_output_last_1 = all_hidden_states[-1][:, 0]
        pooled_output_last_2 = all_hidden_states[-2][:, 0]
        pooled_output_last_3 = all_hidden_states[-3][:, 0]
        pooled_output_last_4 = all_hidden_states[-4][:, 0]

        logits1 = self.classifier1(pooled_output_last_1)
        logits2 = self.classifier2(pooled_output_last_2)
        logits3 = self.classifier3(pooled_output_last_3)
        logits4 = self.classifier4(pooled_output_last_4)

        logits = logits1 + logits2 + logits3 + logits4
        return logits


class BertEnsambleForSequenceClassification(nn.Module):
    def __init__(self, config, bert):
        super().__init__()

        self.num_labels = config.num_labels
        self.config = config

        self.bert = bert
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
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
            output_hidden_states=True,
        )

        # pooled_output = outputs[1]
        hidden_states = torch.stack(outputs[2][-3:])  # 取后三层hidden state
        pooled_output = torch.mean(
            hidden_states[:, :, 0], dim=0)  # 对后三层pooler output求均值

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
