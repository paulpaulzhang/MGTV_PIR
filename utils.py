# coding:utf-8

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import os
import random
import numpy as np
import torch
import math
from collections import defaultdict

from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from transformers import AdamW


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class CosineAnnealingWarmupSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(CosineAnnealingWarmupSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return ((1 + math.cos(step * math.pi / self.t_total)) / 2)


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


class FGM(object):
    """
    基于FGM算法的攻击机制

    Args:
        module (:obj:`torch.nn.Module`): 模型

    Examples::

        >>> # 初始化
        >>> fgm = FGM(module)
        >>> for batch_input, batch_label in data:
        >>>     # 正常训练
        >>>     loss = module(batch_input, batch_label)
        >>>     loss.backward() # 反向传播，得到正常的grad
        >>>     # 对抗训练
        >>>     fgm.attack() # 在embedding上添加对抗扰动
        >>>     loss_adv = module(batch_input, batch_label)
        >>>     loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        >>>     fgm.restore() # 恢复embedding参数
        >>>     # 梯度下降，更新参数
        >>>     optimizer.step()
        >>>     optimizer.zero_grad()

    Reference:
        [1]  https://zhuanlan.zhihu.com/p/91269728
    """

    def __init__(self, module):
        self.module = module
        self.backup = {}

    def attack(
        self,
        epsilon=1.,
        emb_name='word_embeddings'
    ):
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(
        self,
        emb_name='word_embeddings'
    ):
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD(object):
    """
    基于PGD算法的攻击机制

    Args:
        module (:obj:`torch.nn.Module`): 模型

    Examples::

        >>> pgd = PGD(module)
        >>> K = 3
        >>> for batch_input, batch_label in data:
        >>>     # 正常训练
        >>>     loss = module(batch_input, batch_label)
        >>>     loss.backward() # 反向传播，得到正常的grad
        >>>     pgd.backup_grad()
        >>>     # 对抗训练
        >>>     for t in range(K):
        >>>         pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
        >>>         if t != K-1:
        >>>             optimizer.zero_grad()
        >>>         else:
        >>>             pgd.restore_grad()
        >>>         loss_adv = module(batch_input, batch_label)
        >>>         loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        >>>     pgd.restore() # 恢复embedding参数
        >>>     # 梯度下降，更新参数
        >>>     optimizer.step()
        >>>     optimizer.zero_grad()

    Reference:
        [1]  https://zhuanlan.zhihu.com/p/91269728
    """

    def __init__(self, module):
        self.module = module
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(
        self,
        epsilon=1.,
        alpha=0.3,
        emb_name='emb.',
        is_first_attack=False
    ):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class AWP:
    def __init__(
        self,
        model,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        adv_step=1,
    ):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}

    def attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_kl_loss(p, q, pad_mask=None):

    p_loss = F.kl_div(F.softmax(p),
                      F.softmax(q), reduction='none')
    q_loss = F.kl_div(F.softmax(q),
                      F.softmax(p), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


def get_default_bert_optimizer(
    module,
    args,
    eps: float = 1e-6,
    correct_bias: bool = True,
):
    model_param = list(module.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]

    bert_param_optimizer = []
    lstm_param_optimizer = []
    classifier_param_optimizer = []

    for name, param in model_param:
        space = name.split('.')
        if 'bert' in space[0]:
            bert_param_optimizer.append((name, param))
        elif 'bilstm' in space[0]:
            lstm_param_optimizer.append((name, param))
        elif 'classifier' in space[0]:
            classifier_param_optimizer.append((name, param))

    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, 'lr': args.lr},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, 'lr': args.lr},

        {"params": [p for n, p in lstm_param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, 'lr': args.lstm_lr},
        {"params": [p for n, p in lstm_param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, 'lr': args.lstm_lr},

        {"params": [p for n, p in classifier_param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, 'lr': args.clf_lr},
        {"params": [p for n, p in classifier_param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, 'lr': args.clf_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr,
                      eps=eps,
                      correct_bias=correct_bias,
                      weight_decay=args.weight_decay)
    return optimizer


class Logs:
    def __init__(self, path) -> None:
        self.path = path
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write('')

    def write(self, content):
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(content)


def metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    return f1, accuracy, recall
