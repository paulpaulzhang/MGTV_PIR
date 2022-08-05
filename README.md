## 第三届马栏山杯邀请赛-商品意图识别 top3 方案分享
#### 1.比赛简介
官方提供两个文件，goods_data.csv是商品名称数据，query_data.csv是用户query数据，共39470条。其中label是该文本内容的商品购买意图，已进行了id化处理。

典型的文本分类问题，使用Macro F1作为指标。

#### 2.数据清洗

在比赛初期通过观察训练集数据发现数据包含很多“？！”等特殊符号，经过测试发现剔除此类符号可以上小分。

训练集与测试集同样包含很多类似于“拼多多砍一刀”的文本，并且在训练集中类别都为0类，没有训练的意义，就删除了，最后把测试集中将此类文本的类别人工修改为0类。

数据中也出现纯链接地址文本，这种文本删除后对分数没有影响，甚至还高了一些。

#### 3.模型结构

我们使用uer_large预训练模型 [junnyu/uer_large · Hugging Face](https://huggingface.co/junnyu/uer_large)

模型采用bert+bilstm结构，数据经过bert后获取bert最后一层的输出并作为bilstm的输入，最后将bilstm的输出的每一个句子的第一个词向量拿出([CLS])，经过一层线性层进行分类。尝试过bert后四层求均值融合、最后一层求均值作为logits，但都不如最简单的[CLS]向量好使。

#### 4.训练过程

在训练过程中使用了ema，awp，分层学习率，warmup。

使用五折训练，最后每一折的logits融合。

#### 5.预训练

A榜时使用query_data,goods_data,test_a作为预训练语料预训练40 epoch，batch_size 为64，在B榜提交五折融合成绩为0.7628。

B榜时使用的预训练语料在A榜基础上加入了test_b（将0.7628中预测为0的的文本删除，剩余2w+文本），其余参数不变，B榜提交成绩为0.7641，也就是最终成绩。

#### 6.杂谈

这次比赛的成绩简直就是过山车，A榜和B榜的gap将近0.02，属实离谱，赛后复盘觉得可能是预训练起作用或者我们训练的模型刚好适合B榜数据😂。但在复现的时候成绩一直复现不出来，由于第一次使用docker，对docker操作很陌生，以为是环境的问题，整了两天也没结果，最后无奈放弃，使用复现成绩作为最终成绩(0.7606)，最离谱的是榜单出来的那天晚上，我就发现推理时没开model.eval，dropout导致结果无法复现，当场泪崩！

这里要说一下，网上说环境、torch 版本对最后结果影响很大，但在我找问题的过程中，发现这好像也不会引起太大的分数差，我在不同环境测试，结果都差不多，反而是 model.eval，在关闭后，不同机器上似乎会有很大的分差，这也是我第一次知道 model.eval关闭后，同一台机器上的结果是可以复现的，但换一台机器就会有很大的分差，完全无法复现。之前一直以为model.eval关闭后，推理的结果每一次都会不同，导致找问题时压根就没往这个方向想。

算是一次经验积累吧，只不过代价比较大😂