import torch


def predict(text, tokenizer, bert_mlm):
    """
    文章を入力として受け、BERTが予測した文章を出力
    """
    # 符号化
    encoding, spans = tokenizer.encode_plus_untagged(
        text, return_tensors='pt'
    ) 
    #encoding = { k: v.cuda() for k, v in encoding.items() }
    encoding = { k: v for k, v in encoding.items() }

    # ラベルの予測値の計算
    with torch.no_grad():
        output = bert_mlm(**encoding)
        scores = output.logits
        labels_predicted = scores[0].argmax(-1).cpu().numpy().tolist()

    # ラベル列を文章に変換
    predict_text = tokenizer.convert_bert_output_to_text(
        text, labels_predicted, spans
    )

    return predict_text