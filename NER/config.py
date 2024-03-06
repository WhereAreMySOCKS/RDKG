import json

import torch

model_dict = {
    'bert': ('transformers.BertTokenizer',
             'transformers.BertModel',
             'transformers.BertConfig',
             'my_bert/bert_base'  # 使用模型
             ),

    'roberta_base': (
        'transformers.BertTokenizer',
        'transformers.BertModel',
        'transformers.AutoConfig',
        'my_bert/roberta_base',
    ),
    'roberta_large': (
        'transformers.BertTokenizer',
        'transformers.BertModel',
        'transformers.AutoConfig',
        'my_bert/roberta_large',
    )
}
device = torch.device("cuda:5")
# 是否使用bi-lstm层
is_bilstm = False
# is_bilstm = True

MODEL = 'roberta_base'
# MODEL = 'ernie'
# MODEL = 'roberta_base'
# MODEL = 'roberta_large'

epochs = 100
batch_size = 16
lr = 1e-5  # 学习率
patience = 25  # early stop 不变好 就停止
max_grad_norm = 10.0  # 梯度修剪
target_file = 'models/{}.pth.tar'.format(MODEL)  # 模型存储路径
checkpoint = None  # 设置模型路径  会继续训练
n_nums = None  # 读取csv行数，因为有时候测试需要先少读点 None表示读取所有
freeze_bert_head = False  # freeze bert提取特征部分的权重

# 切换任务时 数据配置
csv_rows = ['raw_sen', 'label']  # csv的行标题，文本 和 类（目前类必须是列表）

dir_name = ''
train_file = f"data/train_BIO.csv"
dev_file = f"data/dev_BIO.csv"
test_file = f"data/test_BIO.csv"
csv_encoding = 'utf-8'
test_pred_out = f"data/test_data_predict.csv"
json_dict = f'data/label_2_id.json'


PREFIX = ''
max_seq_len = 512
ignore_pad_token_for_loss = True
overwrite_cache = None


use_crf = True

with open(json_dict, 'r', encoding='utf-8') as f:
    dict_ = json.load(f)
num_labels = len(dict_)
print(f"num_labels 是{num_labels}")
