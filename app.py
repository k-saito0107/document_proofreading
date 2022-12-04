import tkinter as tk
from tkinter import scrolledtext
from utils.article_collection import collection
from utils.bert_model import BertForMaskedLM_pl
from utils.tokenizer import SC_tokenizer


# 日本語の事前学習済みモデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
best_model_path = './weights/epoch=3-step=18740-v1.ckpt'




def main(tokenizer, model):
    #ウィンドウを作
    root = tk.Tk()
    
    #ウィンドウサイズを指定
    root.geometry("700x400")

    #ウィンドウタイトルを指定
    root.title('文章校正')

    #テキストボックス作成
    st1 = tk.Text(
        root,
        width=40,
        height=20
    )
    
    #ウィジェットを配置
    st1.pack(fill = 'x', padx=10, side = 'left')

    #ScrolledTextウィジェットを作成
    '''
    st2= scrolledtext.ScrolledText(
        root, 
        width=45, 
        height=40,
        font=("Helvetica",
        10))
    '''
    st2 = tk.Text(
        root,
        width=40,
        height=20
    )
    #ウィジェットを配置
    st2.pack(fill = 'x', padx=10, side = 'left')


    def push_botton():
        inputs_text = st1.get('1.0', 'end')
        result_text = collection(inputs_text, tokenizer, model)
        #st2.delete(0, tk.END)
        st2.insert(tk.END, result_text)
        #print(result_text)

    # ボタン作成
    btn = tk.Button(root, text='変換', command=push_botton() ,width=14)
    btn.pack(side='left')
    #ウィンドウ表示継続
    root.mainloop()

if __name__ == '__main__':
    tokenizer = SC_tokenizer.from_pretrained(MODEL_NAME)
    model = BertForMaskedLM_pl.load_from_checkpoint(best_model_path)##best_model_path

    main(tokenizer, model)