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


def clean_invite(text):
    tag_pattern = re.compile('(\[|\|【￥)(.*?)(|\]|\】\￥)', re.S)
    text = re.sub(tag_pattern, '', text)
    return text.strip()


def text_enchance(text):
    text = text.strip(' ')
    text = clean_url(text)
    text = clean_invite(text)
    text = re.sub("[？！\r\n]", "", text)
    # text = text.replace(' ', ',')
    return text
