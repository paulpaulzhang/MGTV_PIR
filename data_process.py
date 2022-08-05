import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm
import random
import re


def clean_url(text):
    sentences = text.split(' ')
    # 处理http://类链接
    url_pattern = re.compile(
        r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-)*\b', re.S)
    # 处理无http://类链接
    domain_pattern = re.compile(r'(\b)*(.*?)\.(com|cn)')
    if len(sentences) > 0:
        result = []
        for item in sentences:
            text = re.sub(url_pattern, '', item)
            text = re.sub(domain_pattern, '', text)
            result.append(text)
        return ' '.join(result)
    else:
        return re.sub(url_pattern, '', sentences)


def clean_tag(text):
    tag_pattern = re.compile('[【￥#/(（)](.*?)[】￥#/)）]', re.S)
    text = re.sub(tag_pattern, '', text)
    return text.strip()


def text_enchance(text):
    text = clean_url(text)
    text = re.sub('[？！\r\n\']', '', text)
    text = ' '.join(text.split())
    if '复制' in text:
        text = ''
    return text


def clean_blank(text):
    text = text.replace(',', ' ')
    text = text.split(' ')
    text = [word for word in text if word != '']
    text = ','.join(text)
    return text


def word_select(text):
    words = text.split(',')
    random.shuffle(words)
    return ','.join(words[:random.randint(1, max(1, len(words)-1))])


def random_select(df, repeat_num=1):
    random_sentences = []
    for item in tqdm(df.itertuples(), total=len(df)):
        text = item[2]
        label = item[3]
        random_sentences.extend([[word_select(text), label]
                                 for _ in range(repeat_num)])
    return random_sentences


def generate_sentences(words, gen_max_len):
    random.shuffle(words)
    return ','.join(words[:random.randint(1, gen_max_len)])


def redistribution(df, gen_num_by_label=100, gen_max_len=6):
    label_df = df[['text', 'label']].groupby(by='label',
                                             as_index=False)['text'].apply(lambda x: ','.join(x))
    label_df['words'] = label_df['text'].apply(
        lambda x: list(set(x.split(','))))
    redistribution_sentences = []
    for item in tqdm(label_df.itertuples(), total=len(label_df)):
        label = item[1]
        words = item[3]
        redistribution_sentences.extend(
            [generate_sentences(words, gen_max_len), label]
            for _ in range(gen_num_by_label))
    return redistribution_sentences


if __name__ == '__main__':
    goods_data = pd.read_csv('../data/a_dataset/goods_data.csv')
    goods_data['text'] = goods_data['text'].apply(clean_blank)

    random_df = pd.DataFrame(random_select(
        goods_data), columns=['text', 'label'])
    redistribution_df = pd.DataFrame(redistribution(goods_data),
                                     columns=['text', 'label'])

    extend_data_df = pd.concat(
        [random_df, redistribution_df], axis=0).reset_index(drop=True)
    extend_data_df = extend_data_df.drop_duplicates()
    extend_data_df = shuffle(extend_data_df).reset_index(drop=True)
    extend_data_df.to_csv(
        '../data/a_dataset/extend_data.csv', index_label='id')
