# How to get subtask

## step 1: download the imagenet 1k
要求以下面的格式进行存放
```bash
imagenet
|____train
|________class1
|__________img1
|__________img2
|________class2
|__________img1
|__________img2
|____val
|________class1
|__________img1
|__________img2
|________class2
|__________img1
|__________img2
```

## step 2: download nltk
创建一个python环境，安装nltk和解压wordnet
```bash
pip install nltk
unzip wordnet.zip -d /usr/local/share/nltk_data/corpora
```

## step 3：修改python文件中的imagenet路径
```python
image_net_path = '/home/zhenfeng/project/swintf/imagenet_1k/imagenet/'
```
修改上述语句到对应的数据集位置即可，之后运行python文件就可以进行subtask的获取