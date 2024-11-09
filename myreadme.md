# 我自己用的环境

用的4090的卡，
sudo wget -P /root/autodl-tmp/sru https://download.pytorch.org/whl/cu100/torch-1.0.1-cp37-cp37m-linux_x86_64.whl
先用这个命令将torch下载到自己的电脑上，然后用pip去安装


pip install matplotlib==3.0.0

pip install numpy==1.16.4 scipy==1.2.1

妈的，最后用的还是cpu，GPU的环境配不好，对CUDA有要求好像。


# 修改
我将main里面的-n换成了-sruname，![img.png](img%2Fimg.png)注意框起来的地方都要配置，然后画箭头的那里面把-n改成-sruname，其他不变

主要是因为conda在运行命令的时候也有一个参数叫-n会产生歧义


**`sru.py`**
改了一句
```python
stop_time = min(start_time + blk_size - 1, numTotalSamples - 1)
```

`main2.py`
是我们的主函数
```python
    elif (dataset == 'lorenz'):
        A = [0.0, 0.01, 0.1, 0.99];
        dim_iid_stats = 10
        dim_rec_stats = 10
        dim_final_stats = 10
        dim_rec_stats_feedback = 10
        # batchSize = 250
        # batchSize = 200
        # batchSize = 500
        batchSize = 1000
        blk_size = int(batchSize / 2)
        numBatches = int(numTotalSamples / batchSize)
```
这个地方的barchsize也要改掉。




# 目录说明
SRU_for_GCI/    
│
├── crossval/    #  项目本身的数据集
│   └── logs/    #  
│
├── data/    # 项目本身的数据集 
│   ├── dream3/    #  
│   ├── lorenz96/    #  
│   ├── netsim/    #  
│   ├── var/    #  
│   │   ├── Dream3TensorData/    #  
│   │   └── TrueGeneNetworks/    #  
│   └── logs/    #  
│
├── datasets/    #  姚牧云师兄的数据集
│
├── img/    #  本md的图片路径
│
├── models/    #  项目的模型文件
│
├── my_dataset/    #  蒋文睿自己的生成的数据
│   ├── lorenz96/    #  
│   └── var/    #  
│
└── utils/    #  
