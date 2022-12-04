FROM ubuntu:20.04

RUN apt-get update && apt-get install -y --no-install-recommends wget build-essential libreadline-dev \ 
    libncursesw5-dev libssl-dev libsqlite3-dev libgdbm-dev libbz2-dev liblzma-dev zlib1g-dev uuid-dev libffi-dev libdb-dev

#任意バージョンのpython install
RUN wget --no-check-certificate https://www.python.org/ftp/python/3.9.5/Python-3.9.5.tgz \
    && tar -xf Python-3.9.5.tgz \
    && cd Python-3.9.5 \
    && ./configure --enable-optimizations\
    && make \
    && make install

#サイズ削減のため不要なものは削除
RUN apt-get autoremove -y
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade pip
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip install torch torchvision
#RUN pip install pip install tkinter-page
#RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers==4.18.0 fugashi==1.1.0 ipadic==1.0.0 pytorch-lightning==1.6.1

