from transformers import (AutoModel, AutoModelForMaskedLM,
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)
from data_process import text_enchance
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

goods_data = pd.read_csv('../data/a_dataset/goods_data.csv')
query_data = pd.read_csv('../data/a_dataset/query_data.csv')
test_data = pd.read_csv('../data/a_dataset/test_a.csv')
df = pd.concat([goods_data, query_data, test_data], axis=0)
df['text'] = df['text'].apply(text_enchance)
df['text'] = df['text'].apply(lambda x: x.replace(' ', '\002'))
df = df.drop(df[df['text'] == ''].index)


with open('../data/pretrain_mlm_data.txt', 'w', encoding='utf-8') as f:
    for sentence in df['text'].tolist():
        f.write(sentence + '\n')

model_name = '../pretrain_model/uer_large/'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="../data/pretrain_mlm_data.txt",
    block_size=64)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./checkpoint/uer_pretrain_model",
    overwrite_output_dir=True,
    num_train_epochs=40,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    evaluation_strategy='steps',
    save_total_limit=1,
    gradient_accumulation_steps=1,
    eval_steps=1000,
    save_steps=1000,
    greater_is_better=False,
    load_best_model_at_end=True,
    prediction_loss_only=True,
    remove_unused_columns=False,
    report_to="none",
    learning_rate=2e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=train_dataset)


trainer.train()
trainer.save_model(f'./checkpoint/uer_pretrain_model')
tokenizer.save_pretrained('./checkpoint/uer_pretrain_model')
