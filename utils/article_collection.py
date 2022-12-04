import random
import unicodedata

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForMaskedLM
import pytorch_lightning as pl

from utils.bert_model import BertForMaskedLM_pl
from utils.predict_miss import predict
from utils.tokenizer import SC_tokenizer


# 日本語の事前学習済みモデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
best_model_path = './weights/epoch=3-step=18740-v1.ckpt'


def collection(text, tokenizer, model):
    #text = '私はパソコンを勝った。'
    #text = '私はパソコンを打った'
    tokenizer = SC_tokenizer.from_pretrained(MODEL_NAME)
    model = BertForMaskedLM_pl.load_from_checkpoint(best_model_path)##best_model_path
    bert_mlm = model.bert_mlm#.cuda()
    predict_text = predict(text, tokenizer, bert_mlm)
    #print(predict_text)
    return predict_text