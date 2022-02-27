#  基于PaddleX-YoloV3检测装甲板实现自瞄 


## 项目背景简介

> 全国大学生机器人大赛RoboMaster 机甲大师对抗赛，侧重参赛队员对理工学科的综合应用与工程实践能力，充分融合了“机器视觉”、“嵌入式系统设计”、“机械控制”、“惯性导航”、“人机交互”等众多机器人相关技术学科，同时创新性的将电竞呈现方式与机器人竞技相结合，使机器人对抗更加直观激烈，吸引众多的科技爱好者与社会公众的广泛关注和参与。
 
 该项目使用**PaddleX**提供的YOLOv3模型检测装甲板实现自瞄 

## 目录：
0. 解压数据集unzip；
1. 安装PaddleX；
2. 准备装甲板数据集；
3. 生成训练所需文件；
4. 设置图像数据预处理和数据增强模块；
5. 读取数据集；
6. 定义模型并开始训练；
7. 评估模型性能；
8. 保存模型；
9. 总结

## 最终效果：

![al3ZiF.jpg](https://s1.ax1x.com/2020/07/31/al3ZiF.jpg)
![al3kZV.jpg](https://s1.ax1x.com/2020/07/31/al3kZV.jpg)
![al3EIU.jpg](https://s1.ax1x.com/2020/07/31/al3EIU.jpg)
![al3iq0.jpg](https://s1.ax1x.com/2020/07/31/al3iq0.jpg)

## 0. 解压数据集unzip（只需运行一次）


```python
!unzip /home/aistudio/data/data46309/rmcvdata.zip -d /home/aistudio/work/rmcvdata/
```

    Archive:  /home/aistudio/data/data46309/rmcvdata.zip
    replace /home/aistudio/work/rmcvdata/roco_train/00000001.jpg? [y]es, [n]o, [A]ll, [N]one, [r]ename: ^C


## 1. 安装PaddleX


```python
!pip install pycocotools
!pip install lxml
!pip install paddlex -i https://mirror.baidu.com/pypi/simple
```

    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Collecting pycocotools
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/75/5c/ac61ea715d7a89ecc31c090753bde28810238225ca8b71778dfe3e6a68bc/pycocotools-2.0.4.tar.gz (106 kB)
         |████████████████████████████████| 106 kB 6.0 MB/s            
    [?25h  Installing build dependencies ... [?25ldone
    [?25h  Getting requirements to build wheel ... [?25ldone
    [?25h  Preparing metadata (pyproject.toml) ... [?25ldone
    [?25hRequirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools) (1.19.5)
    Requirement already satisfied: matplotlib>=2.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools) (2.2.3)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (1.1.0)
    Requirement already satisfied: six>=1.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (1.16.0)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (2.8.2)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (2019.3)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (3.0.7)
    Requirement already satisfied: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib>=2.1.0->pycocotools) (56.2.0)
    Building wheels for collected packages: pycocotools
      Building wheel for pycocotools (pyproject.toml) ... [?25ldone
    [?25h  Created wheel for pycocotools: filename=pycocotools-2.0.4-cp37-cp37m-linux_x86_64.whl size=273801 sha256=54afad761aa5299e1611832c152ea191ef20a24fe563fcc0175426311aa0be2f
      Stored in directory: /home/aistudio/.cache/pip/wheels/c0/01/5f/670dfd20204fc9cc6bf843db4e014acb998f411922e3abc49f
    Successfully built pycocotools
    Installing collected packages: pycocotools
    Successfully installed pycocotools-2.0.4
    [33mWARNING: You are using pip version 21.3.1; however, version 22.0.3 is available.
    You should consider upgrading via the '/opt/conda/envs/python35-paddle120-env/bin/python -m pip install --upgrade pip' command.[0m
    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Collecting lxml
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8d/63/03f25363b26fa27a733d920554d73e34390830b3b5c012d7a9f721d1dc2d/lxml-4.8.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_24_x86_64.whl (6.4 MB)
         |████████████████████████████████| 6.4 MB 4.4 MB/s            
    [?25hInstalling collected packages: lxml
    Successfully installed lxml-4.8.0
    [33mWARNING: You are using pip version 21.3.1; however, version 22.0.3 is available.
    You should consider upgrading via the '/opt/conda/envs/python35-paddle120-env/bin/python -m pip install --upgrade pip' command.[0m
    Looking in indexes: https://mirror.baidu.com/pypi/simple
    Collecting paddlex
      Downloading https://mirror.baidu.com/pypi/packages/ca/03/b401c6a34685aa698e7c2fbcfad029892cbfa4b562eaaa7722037fef86ed/paddlex-2.1.0-py3-none-any.whl (1.6 MB)
         |████████████████████████████████| 1.6 MB 7.8 MB/s            
    [?25hRequirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.1.1.26)
    Collecting shapely>=1.7.0
      Downloading https://mirror.baidu.com/pypi/packages/9d/4d/4b0d86ed737acb29c5e627a91449470a9fb914f32640db3f1cb7ba5bc19e/Shapely-1.8.1.post1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (2.0 MB)
         |████████████████████████████████| 2.0 MB 59.9 MB/s            
    [?25hRequirement already satisfied: scipy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (1.6.3)
    Collecting motmetrics
      Downloading https://mirror.baidu.com/pypi/packages/9c/28/9c3bc8e2a87f4c9e7b04ab72856ec7f9895a66681a65973ffaf9562ef879/motmetrics-1.2.0-py3-none-any.whl (151 kB)
         |████████████████████████████████| 151 kB 64.3 MB/s            
    [?25hRequirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.27.0)
    Requirement already satisfied: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.4.4)
    Requirement already satisfied: flask-cors in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (3.0.8)
    Requirement already satisfied: pycocotools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (2.0.4)
    Collecting visualdl>=2.2.2
      Downloading https://mirror.baidu.com/pypi/packages/87/c8/10d0d24822637d8e5493a73ad118640530195e45b1c71ae0e60606ff5f0e/visualdl-2.2.3-py3-none-any.whl (2.7 MB)
         |████████████████████████████████| 2.7 MB 34.0 MB/s            
    [?25hCollecting scikit-learn==0.23.2
      Downloading https://mirror.baidu.com/pypi/packages/f4/cb/64623369f348e9bfb29ff898a57ac7c91ed4921f228e9726546614d63ccb/scikit_learn-0.23.2-cp37-cp37m-manylinux1_x86_64.whl (6.8 MB)
         |████████████████████████████████| 6.8 MB 17.5 MB/s            
    [?25hCollecting lap
      Downloading https://mirror.baidu.com/pypi/packages/bf/64/d9fb6a75b15e783952b2fec6970f033462e67db32dc43dfbb404c14e91c2/lap-0.4.0.tar.gz (1.5 MB)
         |████████████████████████████████| 1.5 MB 34.3 MB/s            
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hCollecting paddleslim==2.2.1
      Downloading https://mirror.baidu.com/pypi/packages/0b/dc/f46c4669d4cb35de23581a2380d55bf9d38bb6855aab1978fdb956d85da6/paddleslim-2.2.1-py3-none-any.whl (310 kB)
         |████████████████████████████████| 310 kB 36.6 MB/s            
    [?25hRequirement already satisfied: chardet in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (3.0.4)
    Requirement already satisfied: openpyxl in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (3.0.5)
    Requirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.1.2)
    Requirement already satisfied: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddleslim==2.2.1->paddlex) (2.2.3)
    Requirement already satisfied: pyzmq in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddleslim==2.2.1->paddlex) (22.3.0)
    Requirement already satisfied: pillow in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddleslim==2.2.1->paddlex) (8.2.0)
    Requirement already satisfied: numpy>=1.13.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn==0.23.2->paddlex) (1.19.5)
    Requirement already satisfied: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn==0.23.2->paddlex) (0.14.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn==0.23.2->paddlex) (2.1.0)
    Requirement already satisfied: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.2.2->paddlex) (1.1.5)
    Requirement already satisfied: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.2.2->paddlex) (1.1.1)
    Requirement already satisfied: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.2.2->paddlex) (1.16.0)
    Requirement already satisfied: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.2.2->paddlex) (4.0.1)
    Requirement already satisfied: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.2.2->paddlex) (0.8.53)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.2.2->paddlex) (2.24.0)
    Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.2.2->paddlex) (3.14.0)
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.2.2->paddlex) (1.21.0)
    Requirement already satisfied: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.2.2->paddlex) (0.7.1.1)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.2.2->paddlex) (1.0.0)
    Collecting pytest-benchmark
      Downloading https://mirror.baidu.com/pypi/packages/2c/60/423a63fb190a0483d049786a121bd3dfd7d93bb5ff1bb5b5cd13e5df99a7/pytest_benchmark-3.4.1-py2.py3-none-any.whl (50 kB)
         |████████████████████████████████| 50 kB 16.7 MB/s            
    [?25hCollecting xmltodict>=0.12.0
      Downloading https://mirror.baidu.com/pypi/packages/28/fd/30d5c1d3ac29ce229f6bdc40bbc20b28f716e8b363140c26eff19122d8a5/xmltodict-0.12.0-py2.py3-none-any.whl (9.2 kB)
    Collecting flake8-import-order
      Downloading https://mirror.baidu.com/pypi/packages/ab/52/cf2d6e2c505644ca06de2f6f3546f1e4f2b7be34246c9e0757c6048868f9/flake8_import_order-0.18.1-py2.py3-none-any.whl (15 kB)
    Collecting pytest
      Downloading https://mirror.baidu.com/pypi/packages/38/93/c7c0bd1e932b287fb948eb9ce5a3d6307c9fc619db1e199f8c8bc5dad95f/pytest-7.0.1-py3-none-any.whl (296 kB)
         |████████████████████████████████| 296 kB 26.4 MB/s            
    [?25hRequirement already satisfied: et-xmlfile in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from openpyxl->paddlex) (1.0.1)
    Requirement already satisfied: jdcal in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from openpyxl->paddlex) (1.4.1)
    Requirement already satisfied: importlib-metadata<4.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.2.2->paddlex) (4.2.0)
    Requirement already satisfied: pycodestyle<2.9.0,>=2.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.2.2->paddlex) (2.8.0)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.2.2->paddlex) (0.6.1)
    Requirement already satisfied: pyflakes<2.5.0,>=2.4.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.2.2->paddlex) (2.4.0)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.2.2->paddlex) (0.16.0)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.2.2->paddlex) (7.0)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.2.2->paddlex) (1.1.0)
    Requirement already satisfied: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.2.2->paddlex) (2.11.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.2.2->paddlex) (2019.3)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.2.2->paddlex) (2.8.0)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->paddleslim==2.2.1->paddlex) (2.8.2)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->paddleslim==2.2.1->paddlex) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->paddleslim==2.2.1->paddlex) (1.1.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->paddleslim==2.2.1->paddlex) (3.0.7)
    Requirement already satisfied: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.2.2->paddlex) (0.18.0)
    Requirement already satisfied: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.2.2->paddlex) (3.9.9)
    Requirement already satisfied: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8-import-order->motmetrics->paddlex) (56.2.0)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.2.2->paddlex) (0.10.0)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.2.2->paddlex) (1.3.0)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.2.2->paddlex) (16.7.9)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.2.2->paddlex) (1.3.4)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.2.2->paddlex) (1.4.10)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.2.2->paddlex) (2.0.1)
    Requirement already satisfied: pluggy<2.0,>=0.12 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pytest->motmetrics->paddlex) (0.13.1)
    Collecting iniconfig
      Downloading https://mirror.baidu.com/pypi/packages/9b/dd/b3c12c6d707058fa947864b67f0c4e0c39ef8610988d7baea9578f3c48f3/iniconfig-1.1.1-py2.py3-none-any.whl (5.0 kB)
    Requirement already satisfied: packaging in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pytest->motmetrics->paddlex) (21.3)
    Collecting tomli>=1.0.0
      Downloading https://mirror.baidu.com/pypi/packages/97/75/10a9ebee3fd790d20926a90a2547f0bf78f371b2f13aa822c759680ca7b9/tomli-2.0.1-py3-none-any.whl (12 kB)
    Requirement already satisfied: attrs>=19.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pytest->motmetrics->paddlex) (21.4.0)
    Collecting py>=1.8.2
      Downloading https://mirror.baidu.com/pypi/packages/f6/f0/10642828a8dfb741e5f3fbaac830550a518a775c7fff6f04a007259b0548/py-1.11.0-py2.py3-none-any.whl (98 kB)
         |████████████████████████████████| 98 kB 12.3 MB/s            
    [?25hCollecting py-cpuinfo
      Downloading https://mirror.baidu.com/pypi/packages/e6/ba/77120e44cbe9719152415b97d5bfb29f4053ee987d6cb63f55ce7d50fadc/py-cpuinfo-8.0.0.tar.gz (99 kB)
         |████████████████████████████████| 99 kB 12.1 MB/s            
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.2.2->paddlex) (1.25.6)
    Requirement already satisfied: idna<3,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.2.2->paddlex) (2.8)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.2.2->paddlex) (2019.9.11)
    Requirement already satisfied: typing-extensions>=3.6.4 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata<4.3->flake8>=3.7.9->visualdl>=2.2.2->paddlex) (4.0.1)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata<4.3->flake8>=3.7.9->visualdl>=2.2.2->paddlex) (3.7.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.1->visualdl>=2.2.2->paddlex) (2.0.1)
    Building wheels for collected packages: lap, py-cpuinfo
      Building wheel for lap (setup.py) ... [?25ldone
    [?25h  Created wheel for lap: filename=lap-0.4.0-cp37-cp37m-linux_x86_64.whl size=1593853 sha256=e7836bbf25bd4c42ea257612f7c577047332b6ef2f3dba7bcd345779ba84715c
      Stored in directory: /home/aistudio/.cache/pip/wheels/95/5f/20/9e2b2cfb8b2bfae5a5374e947511a47c8909e74aaf6d6d4464
      Building wheel for py-cpuinfo (setup.py) ... [?25ldone
    [?25h  Created wheel for py-cpuinfo: filename=py_cpuinfo-8.0.0-py3-none-any.whl size=22245 sha256=3e78cc4a28fc97db08b035a254caa5408658c4f30037fa15c679b392fd65bdf7
      Stored in directory: /home/aistudio/.cache/pip/wheels/9c/57/dd/323247bc3b04fce7bc3fa4c25c106b87f2c13888c240b68723
    Successfully built lap py-cpuinfo
    Installing collected packages: tomli, py, iniconfig, pytest, py-cpuinfo, xmltodict, pytest-benchmark, flake8-import-order, visualdl, shapely, scikit-learn, paddleslim, motmetrics, lap, paddlex
      Attempting uninstall: visualdl
        Found existing installation: visualdl 2.2.0
        Uninstalling visualdl-2.2.0:
          Successfully uninstalled visualdl-2.2.0
      Attempting uninstall: scikit-learn
        Found existing installation: scikit-learn 0.24.2
        Uninstalling scikit-learn-0.24.2:
          Successfully uninstalled scikit-learn-0.24.2
    Successfully installed flake8-import-order-0.18.1 iniconfig-1.1.1 lap-0.4.0 motmetrics-1.2.0 paddleslim-2.2.1 paddlex-2.1.0 py-1.11.0 py-cpuinfo-8.0.0 pytest-7.0.1 pytest-benchmark-3.4.1 scikit-learn-0.23.2 shapely-1.8.1.post1 tomli-2.0.1 visualdl-2.2.3 xmltodict-0.12.0
    [33mWARNING: You are using pip version 21.3.1; however, version 22.0.3 is available.
    You should consider upgrading via the '/opt/conda/envs/python35-paddle120-env/bin/python -m pip install --upgrade pip' command.[0m


## 2. 准备装甲板数据集
我们使用的数据集是 [大连理工大学0bug战队视觉组](https://bbs.robomaster.com/thread-10814-1-1.html) 分享的COCO格式数据

我们先将它转换为VOC格式


```python
from pycocotools.coco import COCO
import os, cv2, shutil
from lxml import etree, objectify
from tqdm import tqdm
from PIL import Image
 
CKimg_dir = 'work/rmcvdata/VOC/images'
CKanno_dir = 'work/rmcvdata/VOC/annotations'
 
 
# 若模型保存文件夹不存在，创建模型保存文件夹，若存在，删除重建
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)
 
 
def save_annotations(filename, objs, filepath):
    annopath = CKanno_dir + "/" + filename[:-3] + "xml"  # 生成的xml文件保存路径
    dst_path = CKimg_dir + "/" + filename
    img_path = filepath
    img = cv2.imread(img_path)
    im = Image.open(img_path)
    if im.mode != "RGB":
        print(filename + " not a RGB image")
        im.close()
        return
    im.close()
    shutil.copy(img_path, dst_path)  # 把原始图像复制到目标文件夹
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('1'),
        E.filename(filename),
        E.source(
            E.database('CKdemo'),
            E.annotation('VOC'),
            E.image('CK')
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(img.shape[2])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose(),
            E.truncated("0"),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[2]),
                E.ymin(obj[3]),
                E.xmax(obj[4]),
                E.ymax(obj[5])
            )
        )
        anno_tree.append(anno_tree2)
    etree.ElementTree(anno_tree).write(annopath, pretty_print=True)
 
 
def showbycv(coco, dataType, img, classes, origin_image_dir, verbose=False):
    filename = img['file_name']
    filepath = os.path.join(origin_image_dir, dataType, filename)
    I = cv2.imread(filepath)
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    for ann in anns:
        name = classes[ann['category_id']]
        if 'bbox' in ann:
            bbox = ann['bbox']
            xmin = (int)(bbox[0])
            ymin = (int)(bbox[1])
            xmax = (int)(bbox[2] + bbox[0])
            ymax = (int)(bbox[3] + bbox[1])
            obj = [name, 1.0, xmin, ymin, xmax, ymax]
            objs.append(obj)
            if verbose:
                cv2.rectangle(I, (xmin, ymin), (xmax, ymax), (255, 0, 0))
                cv2.putText(I, name, (xmin, ymin), 3, 1, (0, 0, 255))
    save_annotations(filename, objs, filepath)
    if verbose:
        cv2.imshow("img", I)
        cv2.waitKey(0)

def catid2name(coco):  # 将名字和id号建立一个字典
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
        # print(str(cat['id'])+":"+cat['name'])
    return classes
 
 
def get_CK5(origin_anno_dir, origin_image_dir, verbose=False):
    dataTypes = ['roco_train', 'roco_val']
    for dataType in dataTypes:
        annFile = '{}.json'.format(dataType)
        annpath = os.path.join(origin_anno_dir, annFile)
        coco = COCO(annpath)
        classes = catid2name(coco)
        imgIds = coco.getImgIds()
        # imgIds=imgIds[0:1000]#测试用，抽取10张图片，看下存储效果
        for imgId in tqdm(imgIds):
            img = coco.loadImgs(imgId)[0]
            showbycv(coco, dataType, img, classes, origin_image_dir, verbose=False)
 
 
def main():
    base_dir = 'work/rmcvdata/VOC'  # step1 这里是一个新的文件夹，存放转换后的图片和标注
    image_dir = os.path.join(base_dir, 'images')  # 在上述文件夹中生成images，annotations两个子文件夹
    anno_dir = os.path.join(base_dir, 'annotations')
    mkr(image_dir)
    mkr(anno_dir)
    origin_image_dir = 'work/rmcvdata'  # step 2原始的coco的图像存放位置
    origin_anno_dir = 'work/rmcvdata'  # step 3 原始的coco的标注存放位置
    verbose = False  # 是否需要看下标记是否正确的开关标记，若是true,就会把标记展示到图片上
    get_CK5(origin_anno_dir, origin_image_dir, verbose)
 
 
if __name__ == "__main__":
    main()
```

    loading annotations into memory...
    Done (t=3.96s)
    creating index...


      0%|          | 17/96953 [00:00<09:32, 169.19it/s]

    index created!


      0%|          | 48/96953 [00:00<08:16, 195.23it/s]100%|██████████| 96953/96953 [05:40<00:00, 284.90it/s]


    loading annotations into memory...
    Done (t=0.19s)
    creating index...


      1%|          | 61/10596 [00:00<00:35, 296.82it/s]

    index created!


      1%|          | 121/10596 [00:00<00:35, 297.03it/s]100%|██████████| 10596/10596 [00:35<00:00, 297.73it/s]


## 3. 生成训练所需文件


```python
import os
import random
import sys
from tqdm import tqdm


root_path = 'work/rmcvdata/VOC'

xmlfilepath = root_path + '/annotations'

txtsavepath = root_path 

if not os.path.exists(root_path):
    print("cannot find such directory: " + root_path)
    exit()

if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

trainval_percent = 0.9
train_percent = 0.8
total_xml = os.listdir(xmlfilepath)[:12000]
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

print("train and val size:", tv)
print("train size:", tr)

ftrainval = open(txtsavepath + '/trainval.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train_list.txt', 'w')
fval = open(txtsavepath + '/val_list.txt', 'w')

for i in tqdm(range(num)):
    name = total_xml[i][:-4] 
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write('images/' + name + '.jpg annotations/' + name + '.xml' + '\n')
        else:
            fval.write('images/' + name + '.jpg annotations/' + name + '.xml' + '\n')
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
```

     13%|█▎        | 1563/12000 [00:00<00:01, 7941.34it/s]

    train and val size: 10800
    train size: 8640


    100%|██████████| 12000/12000 [00:01<00:00, 6199.50it/s]


## 4. 设置图像数据预处理和数据增强模块


```python
# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx


from paddlex.det import transforms
train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250),
    transforms.RandomDistort(),
    transforms.RandomExpand(),
    transforms.RandomCrop(),
    transforms.Resize(target_size=608, interp='RANDOM'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    transforms.Resize(target_size=608, interp='CUBIC'),
    transforms.Normalize(),
])
```

    Your script can be run normally only under PaddleX<2.0.0 but the installed PaddleX version is greater than or equal to 2.0.0, the solution is writen in the link https://github.com/PaddlePaddle/PaddleX/tree/develop/tutorials/train#%E7%89%88%E6%9C%AC%E5%8D%87%E7%BA%A7, please refer to this link ro solve this issue.



    An exception has occurred, use %tb to see the full traceback.


    SystemExit: -1



## 5. 读取数据集

> ### 参数说明：

* **data_dir (str)**: 数据集所在的目录路径。

* **file_list (str)**: 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对data_dir的相对路径）。

* **label_list (str)**: 描述数据集包含的类别信息文件路径。

* **transforms (paddlex.det.transforms)**: 数据集中每个样本的预处理/增强算子，详见paddlex.det.transforms。

* **num_workers (int|str)**：数据集中样本在预处理过程中的线程或进程数。默认为’auto’。当设为’auto’时，根据系统的实际CPU核数设置num_workers: 如果CPU核数的一半大于8，则num_workers为8，否则为CPU核数的一半。

* **buffer_size (int)**: 数据集中样本在预处理过程中队列的缓存长度，以样本数为单位。默认为100。

* **parallel_method (str)**: 数据集中样本在预处理过程中并行处理的方式，支持’thread’线程和’process’进程两种方式。默认为’thread’（Windows和Mac下会强制使用thread，该参数无效）。

* **shuffle (bool)**: 是否需要对数据集中样本打乱顺序。默认为False。


```python
with open('work/rmcvdata/VOC/labels.txt', 'w') as f:
    for v in ['armor_blue', 'armor_red']:
        f.write(v+'\n')

datadir = 'work/rmcvdata/VOC'

train_dataset = pdx.datasets.VOCDetection(
    data_dir= datadir,
    file_list= datadir + '/train_list.txt',
    label_list= datadir + '/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir= datadir ,
    file_list= datadir + '/val_list.txt',
    label_list= datadir + '/labels.txt',
    transforms=eval_transforms)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /tmp/ipykernel_101/187957442.py in <module>
          9     file_list= datadir + '/train_list.txt',
         10     label_list= datadir + '/labels.txt',
    ---> 11     transforms=train_transforms,
         12     shuffle=True)
         13 eval_dataset = pdx.datasets.VOCDetection(


    NameError: name 'train_transforms' is not defined


## 6. 定义模型并开始训练

本文使用DarkNet53作为backbone

> ### 参数说明：  
* **num_classes (int)**: 类别数。默认为80。
* **backbone (str)**: YOLOv3的backbone网络，取值范围为[‘DarkNet53’, ‘ResNet34’, ‘MobileNetV1’, ‘MobileNetV3_large’]。默认为’MobileNetV1’。
* **anchors (list|tuple)**: anchor框的宽度和高度，为None时表示使用默认值 [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]。
* **anchor_masks (list|tuple)**: 在计算YOLOv3损失时，使用anchor的mask索引，为None时表示使用默认值 [[6, 7, 8], [3, 4, 5], [0, 1, 2]]。
* **ignore_threshold (float)**: 在计算YOLOv3损失时，IoU大于ignore_threshold的预测框的置信度被忽略。默认为0.7。
* **nms_score_threshold (float)**: 检测框的置信度得分阈值，置信度得分低于阈值的框应该被忽略。默认为0.01。
* **nms_topk (int)**: 进行NMS时，根据置信度保留的最大检测框数。默认为1000。
* **nms_keep_topk (int)**: 进行NMS后，每个图像要保留的总检测框数。默认为100。
* **nms_iou_threshold (float)**: 进行NMS时，用于剔除检测框IOU的阈值。默认为0.45。
* **label_smooth (bool)**: 是否使用label smooth。默认值为False。
* **train_random_shapes (list|tuple)**: 训练时从列表中随机选择图像大小。默认值为[320, 352, 384, 416, 448, 480, 512, 544, 576, 608]。


```python
net='MobileNetV1'

num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(num_classes=num_classes, backbone=net)
model.train(
    num_epochs=4,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    save_interval_epochs=2,
    save_dir='output/' + net,
    pretrain_weights='IMAGENET'
)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /tmp/ipykernel_97/3565222094.py in <module>
          1 net='MobileNetV1'
          2 
    ----> 3 num_classes = len(train_dataset.labels)
          4 model = pdx.det.YOLOv3(num_classes=num_classes, backbone=net)
          5 model.train(


    NameError: name 'train_dataset' is not defined


## 7. 评估模型性能


```python
model.evaluate(eval_dataset, batch_size=1, epoch_id=None, metric=None, return_details=False)
```


```python
model = pdx.load_model('output/DarkNet53/best_model/')
```


```python
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

image_name = 'work/rmcvdata/roco_train/00000119.jpg'
start = time.time()
result = model.predict(image_name, eval_transforms)
print('infer time:{:.6f}s'.format(time.time()-start))
print('detected num:', len(result))

im = cv2.imread(image_name)
font = cv2.FONT_HERSHEY_SIMPLEX
threshold = 0.1

for value in result:
    xmin, ymin, w, h = np.array(value['bbox']).astype(np.int)
    cls = value['category']
    score = value['score']
    if score < threshold:
        continue
    cv2.rectangle(im, (xmin, ymin), (xmin+w, ymin+h), (0, 255, 0), 4)
    cv2.putText(im, '{:s} {:.3f}'.format(cls, score),
                    (xmin, ymin), font, 0.5, (255, 0, 0), thickness=2)

cv2.imwrite('result.jpg', im)
plt.figure(figsize=(15,12))
plt.imshow(im[:, :, [2,1,0]])
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /tmp/ipykernel_97/487272631.py in <module>
          7 image_name = 'work/rmcvdata/roco_train/00000119.jpg'
          8 start = time.time()
    ----> 9 result = model.predict(image_name, eval_transforms)
         10 print('infer time:{:.6f}s'.format(time.time()-start))
         11 print('detected num:', len(result))


    NameError: name 'model' is not defined


## 8. 保存模型


```python
model.save_model('darknet53-model')
```

9. 总结


```python
# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
!ls /home/aistudio/data
```


```python
# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
!ls /home/aistudio/work
```


```python
# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
!mkdir /home/aistudio/external-libraries
!pip install beautifulsoup4 -t /home/aistudio/external-libraries
```


```python
# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')
```

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
