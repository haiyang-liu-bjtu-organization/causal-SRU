# 我自己用的环境

用的4090的卡，
sudo wget -P /root/autodl-tmp/sru https://download.pytorch.org/whl/cu100/torch-1.0.1-cp37-cp37m-linux_x86_64.whl
先用这个命令将torch下载到自己的电脑上，然后用pip去安装


pip install matplotlib==3.0.0

pip install numpy==1.16.4 scipy==1.2.1




# 修改
我将main里面的-n换成了-sruname，![img.png](img%2Fimg.png)注意框起来的地方都要配置，然后画箭头的那里面把-n改成-sruname，其他不变

主要是因为conda在运行命令的时候也有一个参数叫-n会产生歧义

