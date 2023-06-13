import argparse
import wandb
import torch
import logging
import os

from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from tools import preprocess
from tools.dataset import TextClassifyDataset, collate_fn
from tools.trainer import train, validate
from utils import manual_seed, model_save, setup_logger


# 初始化 logger
logger = setup_logger()


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='TextClassify', help='name of the model')
parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU')
parser.add_argument('--gpu_id', type=int, default=0, help='选择使用GPU')
parser.add_argument('--preprocess', type=bool, default=False, help='预处理')
parser.add_argument('--data_path', type=str, default='data/dataset', help='预处理保存文件夹')
parser.add_argument('--out_path', type=str, default='data/output', help='预处理保存文件夹')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--num_labels', type=int, default=4, help='num_labels')
parser.add_argument('--num_epochs', type=int, default=10, help='num_epochs')
parser.add_argument('--seed', type=int, default=1024, help='num_epochs')
parser.add_argument('--early_stopping_patience', type=float, default=0.003, help='early_stopping_patience')
parser.add_argument('--max_length', type=int, default=256, help='文本最大长度')
parser.add_argument('--bert_path', type=str, default='/root/sunyd/pretrained_models/bert-base-chinese/', help='bert_path')

args = parser.parse_args()

args.cwd = os.getcwd()
logger.info(f'\n{args.__dict__}')

wandb.init(project="党史智能化", name=args.model_name)
wandb.watch_called = False

# device
if args.use_gpu and torch.cuda.is_available():
    device = torch.device('cuda', args.gpu_id)
else:
    device = torch.device('cpu')
logger.info(f'device: {device}')

tokenizer = BertTokenizer.from_pretrained(args.bert_path)
args.tokenizer = tokenizer

# 如果不修改预处理的过程，这一步最好注释掉，不用每次运行都预处理数据一次
if args.preprocess:
    preprocess(args)

train_data_path = os.path.join(args.cwd, args.out_path, 'train.pkl')
valid_data_path = os.path.join(args.cwd, args.out_path, 'valid.pkl')
test_data_path = os.path.join(args.cwd, args.out_path, 'test.pkl')

train_dataset = TextClassifyDataset(train_data_path)
valid_dataset = TextClassifyDataset(valid_data_path)
test_dataset = TextClassifyDataset(test_data_path)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn(args))
valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn(args))
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn(args))


model = BertForSequenceClassification.from_pretrained(args.bert_path, num_labels=args.num_labels)
model.to(device)

wandb.watch(model, log="all")
logger.info(f'\n {model}')

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_dataloader) * args.num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
criterion = None

best_f1, best_epoch = -1, 0
es_loss, es_f1, es_epoch, es_patience, best_es_epoch, best_es_f1, es_path, best_es_path = 1e8, -1, 0, 0, 0, -1, '', ''
train_losses, valid_losses = [], []
for epoch in range(args.num_epochs):
    manual_seed(args.seed + epoch)
    train_loss = train(epoch, model, train_dataloader, optimizer, criterion, device, args)
    valid_f1, valid_loss = validate(epoch, model, valid_dataloader, criterion, device, args)
    scheduler.step(valid_loss)
    
    model_path = model_save(model, epoch, args)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    wandb.log({
            "train_loss":train_loss,
            "valid_loss":valid_loss
        })
    
    wandb.log({
            "valid_f1":valid_f1,
        })
    
    if best_f1 < valid_f1:
            best_f1 = valid_f1
            best_epoch = epoch
    # 使用 valid loss 做 early stopping 的判断标准
    if es_loss > valid_loss:
        es_loss = valid_loss
        es_f1 = valid_f1
        es_epoch = epoch
        es_patience = 0
        es_path = model_path
    else:
        es_patience += 1
        if es_patience >= args.early_stopping_patience:
            best_es_epoch = es_epoch
            best_es_f1 = es_f1
            best_es_path = es_path
    
    
    logger.info(f'best(valid loss quota) early stopping epoch: {best_es_epoch}, '
                f'this epoch macro f1: {best_es_f1:0.4f}')
    logger.info(f'this model save path: {best_es_path}')
    logger.info(f'total {epoch} epochs, best(valid macro f1) epoch: {best_epoch}, '
                f'this epoch macro f1: {best_f1:.4f}')

    logger.info('=====end of training====')
    logger.info('')
    logger.info('=====start test performance====')
    _ , test_loss = validate(-1, model, test_dataloader, criterion, device, args)

    wandb.log({
        "test_loss":test_loss,
    })
    
    logger.info('=====ending====')
            
