FROM continuumio/anaconda3:latest
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install pika  -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install hdfs  -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install pymongo  -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install mlflow -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
RUN conda config --set show_channel_urls yes
ADD ./runtime.py /home/runtime/runtime.py
ADD ./process.py /home/runtime/process.py
ADD ./mongo.py /home/runtime/mongo.py
# ENTRYPOINT ["python","/home/runtime/runtime.py"]

