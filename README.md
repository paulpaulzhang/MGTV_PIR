# MGTV_PIR
## 第三届“马栏山杯”国际音视频算法大赛>邀请赛>商品意图识别
## A榜top8 B榜top2
### 文件说明

- run.py: 代码主入口，包含训练，推理，融合等
  
- pretrain.py,pretrain2.py: 预训练代码
  
- data_process.py: 数据处理代码，只是用了clean_url与text_enhence
  
- task.py:训练过程代码
  
- tokenizer.py: 分词器代码
  
- utils.py:常用工具集合
  
- model/model.py:模型结构代码
  
- model/nezha:未使用
  
- datasets.py: 未使用
  

#### 1.数据清洗

    在比赛初期通过观察训练集数据发现数据包含很多“？！”等特殊符号，经过测试发现剔除此类符号能够提升预测精度，因此对此类符号进行剔除。

    同时，训练集与测试集同样包含很多类似于“拼多多砍一刀”的文本，并且在训练集中类别都为0类，因此对此类文本进行删除，并且在最后的预测结果中将此类文本的类别人工修改为0类。

    同样的，数据中出现纯链接地址文本，此类文本我们认为是无信息文本，所以对其进行剔除。

#### 2.模型结构

    我们使用uer_large预训练模型 [junnyu/uer_large · Hugging Face](https://huggingface.co/junnyu/uer_large)

    模型采用bert+bilstm结构，数据经过bert后获取bert最后一层的输出并作为bilstm的输入，最后将bilstm的输出的每一个句子的第一个词向量拿出([CLS])，经过一层线性层进行分类。

#### 3.训练过程

    在训练过程中使用了ema，awp，分层学习率，warmup

    使用五折训练，最后每一折概率融合

#### 4.预训练

    A榜时使用query_data,goods_data,test_a作为预训练语料预训练40epoch，batch_size为64，结合上述操作，此模型在B榜提交成绩为0.7628

    B榜时使用的预训练语料在A榜基础上加入了test_b（将0.7628中预测为0的的文本删除，剩余2w+文本），其余参数不变，B榜提交成绩为**0.7641**，也就是最终成绩
