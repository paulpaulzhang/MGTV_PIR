from email.policy import strict
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter
import gc
import math
from ark_nlp.dataset.base._sentence_classification_dataset import SentenceClassificationDataset
from ark_nlp.factory.loss_function.focal_loss import FocalLoss
from transformers import BertConfig, BertModel
from ark_nlp.processor.tokenizer.transfomer import SentenceTokenizer
from model.nezha.configuration_nezha import NeZhaConfig
from model.nezha.modeling_nezha import NeZhaModel, NeZhaForSequenceClassification
from tokenizer import BertSpanTokenizer
from utils import WarmupLinearSchedule, seed_everything, get_default_bert_optimizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from task import Task
from tqdm import tqdm
from argparse import ArgumentParser
from model.model import BertForSequenceClassification, BertEnsambleForSequenceClassification, BertBiLSTMForSequenceClassification
from data_process import text_enchance
import pandas as pd
import torch
import os
import warnings


def build_model_and_tokenizer(args, num_labels, is_train=True):
    tokenizer = BertSpanTokenizer(vocab=args.model_name_or_path,
                                  max_seq_len=args.max_seq_len)
    config = NeZhaConfig.from_pretrained(args.model_name_or_path,
                                         num_labels=num_labels)
    if is_train:
        bert = NeZhaModel.from_pretrained(
            args.model_name_or_path, config=config)
        dl_module = BertBiLSTMForSequenceClassification(config, bert)
    else:
        bert = NeZhaModel(config=config)
        dl_module = BertBiLSTMForSequenceClassification(config, bert)
    return tokenizer, dl_module


def train(args):
    train_data_df = pd.read_csv(args.data_path)
    goods_df = pd.read_csv(args.goods_data_path)
    train_data_df = pd.concat([train_data_df, goods_df])
    train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))

    train_data_df['text'] = train_data_df['text'].apply(
        lambda x: text_enchance(x))
    train_data_df = train_data_df.drop(
        train_data_df[(train_data_df['text'] == '')].index)

    train_data_df, dev_data_df = train_test_split(
        train_data_df, test_size=0.1, shuffle=True, random_state=args.seed)

    train_dataset = SentenceClassificationDataset(
        train_data_df, categories=sorted(train_data_df['label'].unique()))
    dev_dataset = SentenceClassificationDataset(
        dev_data_df, categories=train_dataset.categories)

    tokenizer, dl_module = build_model_and_tokenizer(
        args, len(train_dataset.cat2id), is_train=True)

    train_dataset.convert_to_ids(tokenizer)
    dev_dataset.convert_to_ids(tokenizer)

    optimizer = get_default_bert_optimizer(dl_module, args)

    if args.warmup_ratio:
        train_steps = args.num_epochs * \
            int(math.ceil(math.ceil(len(train_dataset) /
                args.batch_size) / args.gradient_accumulation_steps))
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)
    else:
        scheduler = None

    torch.cuda.empty_cache()

    model = Task(
        dl_module, optimizer, 'ce',
        scheduler=scheduler,
        ema_decay=args.ema_decay,
        device=args.device)

    model.fit(args,
              train_dataset,
              dev_dataset,
              epochs=args.num_epochs,
              batch_size=args.batch_size,
              num_workers=args.num_workers,
              save_last_model=False,
              gradient_accumulation_steps=args.gradient_accumulation_steps)


def evaluate(args):
    train_data_df = pd.read_csv(args.data_path)
    goods_df = pd.read_csv(args.goods_data_path)
    train_data_df = pd.concat([train_data_df, goods_df])
    train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))
    train_data_df, dev_data_df = train_test_split(
        train_data_df, test_size=0.1, shuffle=True, random_state=args.seed)

    train_data_df['text'] = train_data_df['text'].apply(
        lambda x: text_enchance(x))
    train_data_df = train_data_df.drop(
        train_data_df[(train_data_df['text'] == '')].index)

    train_dataset = SentenceClassificationDataset(
        train_data_df, categories=sorted(train_data_df['label'].unique()))
    dev_dataset = SentenceClassificationDataset(
        dev_data_df, categories=train_dataset.categories)

    tokenizer, dl_module = build_model_and_tokenizer(
        args, len(train_dataset.cat2id), is_train=False)
    dl_module.load_state_dict(torch.load(args.predict_model), strict=False)

    train_dataset.convert_to_ids(tokenizer)
    dev_dataset.convert_to_ids(tokenizer)

    optimizer = get_default_bert_optimizer(dl_module, args)

    torch.cuda.empty_cache()

    model = Task(
        dl_module, optimizer, 'ce',
        ema_decay=args.ema_decay,
        device=args.device)

    model.id2cat = train_dataset.id2cat
    model.evaluate(dev_dataset, num_workers=args.num_workers,
                   evaluate_batch_size=args.batch_size)


def predict(args):
    train_data_df = pd.read_csv(args.data_path)
    train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))

    test_data_df = pd.read_csv(args.test_file)
    test_data_df['label'] = 1
    test_data_df['label'] = test_data_df['label'].apply(lambda x: str(x))

    train_data_df['text'] = train_data_df['text'].apply(
        lambda x: text_enchance(x))
    train_data_df = train_data_df.drop(
        train_data_df[(train_data_df['text'] == '')].index)
    test_data_df['text'] = test_data_df['text'].apply(
        lambda x: text_enchance(x))
    test_data_df.loc[(test_data_df['text'] == ''), 'text'] = '比赛占位字符'

    test_dataset = SentenceClassificationDataset(
        test_data_df, categories=sorted(train_data_df['label'].unique()))

    tokenizer, model = build_model_and_tokenizer(
        args, len(test_dataset.cat2id), is_train=False)
    model.load_state_dict(torch.load(args.predict_model), strict=False)
    model.to(torch.device(args.device))

    test_dataset.convert_to_ids(tokenizer)

    test_generator = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers)

    y_pred = []

    with torch.no_grad():
        for inputs in tqdm(test_generator):
            inputs['input_ids'] = inputs['input_ids'].to(
                torch.device(args.device))
            inputs['attention_mask'] = inputs['attention_mask'].to(
                torch.device(args.device))
            inputs['token_type_ids'] = inputs['token_type_ids'].to(
                torch.device(args.device))
            inputs['label_ids'] = inputs['label_ids'].to(
                torch.device(args.device))

            outputs = model(**inputs)
            y_pred += torch.argmax(outputs, dim=1).cpu().numpy().tolist()

    os.makedirs(args.save_path, exist_ok=True)
    test_data_df['label'] = [test_dataset.id2cat[label] for label in y_pred]
    test_data_df.loc[(test_data_df['text'] == '比赛占位字符'), 'label'] = 0
    test_data_df['label'] = test_data_df['label'].astype('int')
    test_data_df.to_csv(f'{args.save_path}/results.csv', index=None)


def train_cv(args):
    data_df = pd.read_csv(args.data_path)
    goods_df = pd.read_csv(args.goods_data_path)
    # extend_df = pd.read_csv('../data/a_dataset/extend_data.csv')
    data_df = pd.concat([data_df, goods_df])
    data_df['label'] = data_df['label'].apply(lambda x: str(x))

    data_df['text'] = data_df['text'].apply(lambda x: text_enchance(x))
    data_df = data_df.drop(data_df[(data_df['text'] == '')].index)

    kfold = StratifiedKFold(
        n_splits=args.fold, shuffle=True, random_state=args.seed)
    args.checkpoint = os.path.join(args.checkpoint, args.model_type)
    model_type = args.model_type
    for fold, (train_idx, dev_idx) in enumerate(kfold.split(data_df, data_df['label'])):
        print(f'========== {fold + 1} ==========')
        args.model_type = f'{model_type}-{fold + 1}'

        train_data_df, dev_data_df = data_df.iloc[train_idx], data_df.iloc[dev_idx]
        train_dataset = SentenceClassificationDataset(
            train_data_df, categories=sorted(train_data_df['label'].unique()))
        dev_dataset = SentenceClassificationDataset(
            dev_data_df, categories=train_dataset.categories)

        tokenizer, dl_module = build_model_and_tokenizer(
            args, len(train_dataset.cat2id), is_train=True)

        train_dataset.convert_to_ids(tokenizer)
        dev_dataset.convert_to_ids(tokenizer)

        optimizer = get_default_bert_optimizer(dl_module, args)

        if args.warmup_ratio:
            train_steps = args.num_epochs * \
                int(math.ceil(math.ceil(len(train_dataset) /
                    args.batch_size) / args.gradient_accumulation_steps))
            scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)
        else:
            scheduler = None

        model = Task(
            dl_module, optimizer, 'ce',
            scheduler=scheduler,
            ema_decay=args.ema_decay,
            device=args.device)

        model.fit(args,
                  train_dataset,
                  dev_dataset,
                  epochs=args.num_epochs,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  save_last_model=False,
                  gradient_accumulation_steps=args.gradient_accumulation_steps)

        del model, tokenizer, dl_module, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()


def predict_vote(args):
    train_data_df = pd.read_csv(args.data_path)
    train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))

    test_data_df = pd.read_csv(args.test_file)
    test_data_df['label'] = 1
    test_data_df['label'] = test_data_df['label'].apply(lambda x: str(x))

    train_data_df['text'] = train_data_df['text'].apply(
        lambda x: text_enchance(x))
    train_data_df = train_data_df.drop(
        train_data_df[(train_data_df['text'] == '')].index)
    test_data_df['text'] = test_data_df['text'].apply(
        lambda x: text_enchance(x))
    test_data_df.loc[(test_data_df['text'] == ''), 'text'] = '比赛占位字符'

    test_dataset = SentenceClassificationDataset(
        test_data_df, categories=sorted(train_data_df['label'].unique()))

    tokenizer, _ = build_model_and_tokenizer(
        args, len(test_dataset.cat2id), is_train=False)

    test_dataset.convert_to_ids(tokenizer)
    test_generator = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers)

    args.checkpoint = os.path.join(args.checkpoint, args.model_type)
    model_type = args.model_type

    args.save_path = os.path.join(args.save_path, model_type)
    os.makedirs(args.save_path, exist_ok=True)

    for fold in range(args.fold):
        print(f'========== {fold + 1} ==========')
        args.model_type = f'{model_type}-{fold + 1}'
        args.predict_model = os.path.join(
            args.checkpoint, args.model_type, 'best_model.pth')

        _, model = build_model_and_tokenizer(
            args, len(test_dataset.cat2id), is_train=False)
        model.load_state_dict(torch.load(args.predict_model), strict=False)
        model.to(torch.device(args.device))

        y_pred = []

        with torch.no_grad():
            for inputs in tqdm(test_generator):
                inputs['input_ids'] = inputs['input_ids'].to(
                    torch.device(f'cuda:{args.cuda_device}'))
                inputs['attention_mask'] = inputs['attention_mask'].to(
                    torch.device(f'cuda:{args.cuda_device}'))
                inputs['token_type_ids'] = inputs['token_type_ids'].to(
                    torch.device(f'cuda:{args.cuda_device}'))
                inputs['label_ids'] = inputs['label_ids'].to(
                    torch.device(f'cuda:{args.cuda_device}'))

                outputs = model(**inputs)
                y_pred += torch.argmax(outputs, dim=1).cpu().numpy().tolist()

        os.makedirs(args.save_path, exist_ok=True)
        test_data_df['label'] = [test_dataset.id2cat[label]
                                 for label in y_pred]
        test_data_df.loc[(test_data_df['text'] == '比赛占位字符'), 'label'] = 0
        test_data_df['label'] = test_data_df['label'].astype('int')
        test_data_df.to_csv(os.path.join(
            args.save_path, f'{model_type}-{fold + 1}.csv'), index=None)


def predict_merge(args):
    train_data_df = pd.read_csv(args.data_path)
    train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))

    test_data_df = pd.read_csv(args.test_file)
    test_data_df['label'] = 1
    test_data_df['label'] = test_data_df['label'].apply(lambda x: str(x))

    train_data_df['text'] = train_data_df['text'].apply(
        lambda x: text_enchance(x))
    train_data_df = train_data_df.drop(
        train_data_df[(train_data_df['text'] == '')].index)
    test_data_df['text'] = test_data_df['text'].apply(
        lambda x: text_enchance(x))
    test_data_df.loc[(test_data_df['text'] == ''), 'text'] = '比赛占位字符'

    test_dataset = SentenceClassificationDataset(
        test_data_df, categories=sorted(train_data_df['label'].unique()))

    tokenizer, _ = build_model_and_tokenizer(
        args, len(test_dataset.cat2id), is_train=False)

    test_dataset.convert_to_ids(tokenizer)
    test_generator = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers)

    args.checkpoint = os.path.join(args.checkpoint, args.model_type)
    model_type = args.model_type

    os.makedirs(args.save_path, exist_ok=True)

    y_preds = np.zeros((len(test_dataset), len(test_dataset.cat2id)))

    for fold in range(args.fold):
        print(f'========== {fold + 1} ==========')
        args.model_type = f'{model_type}-{fold + 1}'
        args.predict_model = os.path.join(
            args.checkpoint, args.model_type, 'best_model.pth')

        _, model = build_model_and_tokenizer(
            args, len(test_dataset.cat2id), is_train=False)
        model.load_state_dict(torch.load(args.predict_model), strict=False)
        model.to(torch.device(args.device))

        y_pred = []

        with torch.no_grad():
            for inputs in tqdm(test_generator):
                inputs['input_ids'] = inputs['input_ids'].to(
                    torch.device(args.device))
                inputs['attention_mask'] = inputs['attention_mask'].to(
                    torch.device(args.device))
                inputs['token_type_ids'] = inputs['token_type_ids'].to(
                    torch.device(args.device))
                inputs['label_ids'] = inputs['label_ids'].to(
                    torch.device(args.device))

                outputs = model(**inputs)
                y_pred.append(outputs.cpu().numpy())
        y_pred = np.vstack(y_pred)
        y_preds += y_pred / args.fold

    test_data_df['label'] = [test_dataset.id2cat[id_]
                             for id_ in np.argmax(y_preds, axis=1)]
    test_data_df.loc[(test_data_df['text'] == '比赛占位字符'), 'label'] = 0
    test_data_df['label'] = test_data_df['label'].astype('int')
    test_data_df.to_csv(os.path.join(
        args.save_path, f'results.csv'), index=None)


def vote(args):
    path = [str(p) for p in list(Path(args.vote_path).glob('**/*.csv'))]
    all_labels = []

    out_df = pd.read_csv(path[0])

    for p in path:
        tmp_df = pd.read_csv(p)
        all_labels.append(tmp_df['label'])

    merged_label = []
    for row in zip(*all_labels):
        label = Counter(row).most_common(n=1)
        merged_label.append(label[0][0])

    out_df['label'] = merged_label
    out_df.to_csv(f'{args.save_path}/results_vote.csv', index=None)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model_type', type=str,
                        default='bert_base')
    parser.add_argument('--model_name_or_path', type=str,
                        default='../pretrain_model/chinese-bert-wwm/')

    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoint')
    parser.add_argument('--data_path', type=str,
                        default='../data/a_dataset/query_data.csv')
    parser.add_argument('--goods_data_path', type=str,
                        default='../data/a_dataset/goods_data.csv')
    parser.add_argument('--test_file', type=str,
                        default='../data/a_dataset/test_a.csv')
    parser.add_argument('--save_path', type=str, default='./submit')

    parser.add_argument('--do_predict', action='store_true', default=False)
    parser.add_argument('--do_eval', action='store_true', default=False)
    parser.add_argument('--do_train_cv', action='store_true', default=False)
    parser.add_argument('--do_predict_vote',
                        action='store_true', default=False)
    parser.add_argument('--do_predict_merge',
                        action='store_true', default=False)
    parser.add_argument('--do_vote', action='store_true', default=False)
    parser.add_argument('--predict_model', type=str)

    parser.add_argument('--max_seq_len', type=int, default=64)

    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--lstm_lr', type=float, default=1e-2)
    parser.add_argument('--clf_lr', type=float, default=2e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--early_stopping', type=int, default=1)

    parser.add_argument('--use_fgm', action='store_true', default=True)
    parser.add_argument('--use_pgd', action='store_true', default=False)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)

    parser.add_argument('--adv_k', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--emb_name', type=str, default='word_embeddings.')
    parser.add_argument('--adv_lr', type=int, default=3)  # TODO
    parser.add_argument('--adv_eps', type=int, default=3)  # TODO

    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--extend_save_path', type=str,
                        default='./extend_data/')
    parser.add_argument('--vote_path', type=str,
                        default='./submit/')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    seed_everything(args.seed)

    print(args)

    if args.do_predict:
        predict(args)
    elif args.do_eval:
        evaluate(args)
    elif args.do_train_cv:
        train_cv(args)
    elif args.do_predict_vote:
        predict_vote(args)
    elif args.do_predict_merge:
        predict_merge(args)
    elif args.do_vote:
        vote(args)
    else:
        train(args)
