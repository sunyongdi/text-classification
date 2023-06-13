import logging
import os
from typing import List, Dict
from transformers import BertTokenizer

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils import save_pkl, load_csv

logger = logging.getLogger(__name__)




def sent_to_tokenizer(data, cfg):
    # 文本序列化
    for d in data:
        sent = d['text_a']
        encoded_dict = cfg.tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=cfg.max_length,
                pad_to_max_length=True,
                return_attention_mask=True,
                truncation = True,
            )
        d['inputs'] = encoded_dict
    

    

def preprocess(cfg):
    """
    数据预处理阶段
    """
    logger.info('===== start preprocess data =====')
    
    train_fp = os.path.join(cfg.cwd, cfg.data_path, 'train.csv')
    valid_fp = os.path.join(cfg.cwd, cfg.data_path, 'valid.csv')
    test_fp = os.path.join(cfg.cwd, cfg.data_path, 'test.csv')
    
    logger.info('load raw files...')
    train_data = load_csv(train_fp)
    valid_data = load_csv(valid_fp)
    test_data = load_csv(test_fp)
    
    logger.info(' sent_to_tokenizer...')
    sent_to_tokenizer(train_data, cfg)
    sent_to_tokenizer(valid_data, cfg)
    sent_to_tokenizer(test_data, cfg)

    logger.info('save data for backup...')
    os.makedirs(os.path.join(cfg.cwd, cfg.out_path), exist_ok=True)
    train_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'train.pkl')
    valid_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'valid.pkl')
    test_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'test.pkl')
    save_pkl(train_data, train_save_fp)
    save_pkl(valid_data, valid_save_fp)
    save_pkl(test_data, test_save_fp)

    logger.info('===== end preprocess data =====')


if __name__ == '__main__':
    class Config:
        data_path = '/root/sunyd/code/PartyMind/codes/text_classification/data'
        max_length = 128
        out_path = '/root/sunyd/code/PartyMind/codes/text_classification/output'
    cfg = Config()
    cfg.cwd = os.getcwd()
    tokenizer = BertTokenizer.from_pretrained('/root/sunyd/pretrained_models/bert-base-chinese/')
    cfg.tokenizer = tokenizer
    preprocess_data(cfg)
