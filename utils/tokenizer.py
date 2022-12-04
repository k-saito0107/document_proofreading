import random
from tqdm import tqdm
import unicodedata

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForMaskedLM
import pytorch_lightning as pl

from utils.bert_model import BertForMaskedLM_pl


class SC_tokenizer(BertJapaneseTokenizer):
    
    def encode_plus_tagged(
        self, wrong_text, correct_text, max_length=128
    ):
        """
        ファインチューニング時に使用。
        誤変換を含む文章と正しい文章を入力とし、
        符号化を行いBERTに入力できる形式にする。
        """
        # 誤変換した文章をトークン化し、符号化
        encoding = self(
            wrong_text, 
            max_length=max_length, 
            padding='max_length', 
            truncation=True
        )
        # 正しい文章をトークン化し、符号化
        encoding_correct = self(
            correct_text,
            max_length=max_length,
            padding='max_length',
            truncation=True
        ) 
        # 正しい文章の符号をラベルとする
        encoding['labels'] = encoding_correct['input_ids'] 

        return encoding

    def encode_plus_untagged(
        self, text, max_length=None, return_tensors=None
    ):
        """
        文章を符号化し、それぞれのトークンの文章中の位置も特定しておく。
        """
        # 文章のトークン化を行い、
        # それぞれのトークンと文章中の文字列を対応づける。
        tokens = [] # トークンを追加していく。
        tokens_original = [] # トークンに対応する文章中の文字列を追加していく。
        words = self.word_tokenizer.tokenize(text) # MeCabで単語に分割
        
        for word in words:
            # 単語をサブワードに分割
            tokens_word = self.subword_tokenizer.tokenize(word) 
            tokens.extend(tokens_word)
            if tokens_word[0] == '[UNK]': # 未知語への対応
                tokens_original.append(word)
            else:
                tokens_original.extend([
                    token.replace('##','') for token in tokens_word
                ])
        
        #print(tokens_original)
        #print('---------------------------------------------------------')
        # 各トークンの文章中での位置を調べる。（空白の位置を考慮する）
        position = 0
        spans = [] # トークンの位置を追加していく。
        for token in tokens_original:
            l = len(token)
            while 1:
                if token != text[position:position+l]:
                    position += 1
                else:
                    spans.append([position, position+l])
                    position += l
                    break
        #print(spans)
        #print(tokens)
        # 符号化を行いBERTに入力できる形式にする。
        input_ids = self.convert_tokens_to_ids(tokens) 
        encoding = self.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length' if max_length else False, 
            truncation=True if max_length else False
        )
        sequence_length = len(encoding['input_ids'])
        # 特殊トークン[CLS]に対するダミーのspanを追加。
        spans = [[-1, -1]] + spans[:sequence_length-2] 
        # 特殊トークン[SEP]、[PAD]に対するダミーのspanを追加。
        spans = spans + [[-1, -1]] * ( sequence_length - len(spans) ) 

        # 必要に応じてtorch.Tensorにする。
        if return_tensors == 'pt':
            encoding = { k: torch.tensor([v]) for k, v in encoding.items() }

        return encoding, spans

    def convert_bert_output_to_text(self, text, labels, spans):
        """
        推論時に使用。
        文章と、各トークンのラベルの予測値、文章中での位置を入力とする。
        そこから、BERTによって予測された文章に変換。
        """
        assert len(spans) == len(labels)

        # labels, spansから特殊トークンに対応する部分を取り除く
        labels = [label for label, span in zip(labels, spans) if span[0]!=-1]
        spans = [span for span in spans if span[0]!=-1]

        # BERTが予測した文章を作成
        predicted_text = ''
        position = 0
        for label, span in zip(labels, spans):
            start, end = span
            if position != start: # 空白の処理
                predicted_text += text[position:start]
            predicted_token = self.convert_ids_to_tokens(label)
            predicted_token = predicted_token.replace('##', '')
            predicted_token = unicodedata.normalize(
                'NFKC', predicted_token
            ) 
            predicted_text += predicted_token
            position = end
        
        return predicted_text