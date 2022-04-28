from torch import nn
import torch


class BertForSequenceClassification(nn.Module):
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
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

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
        hidden_states = torch.stack(outputs[2][-3:]) # 取后三层hidden state
        pooled_output = torch.mean(hidden_states[:, :, 0], dim=0) # 对后三层pooler output求均值

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
