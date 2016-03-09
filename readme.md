

论坛垃圾帖分类

scikit-learn + jieba + flask

运行

`pip install -r requirements.txt`
`gunicorn -w 2 -b 127.0.0.1:5000 main:app`


学习

`curl 'http://localhost:5000/train/?content=两个黄鹂鸣翠柳，一行白鹭上青天&type=ham'`
会生成文件`tmpVector.pkl*`缓存HashingVectorizer向量，方便分类时使用，无需每次查数据仓库文件
文件`train_data0.txt``train_data1.txt`,数据仓库，存储已学习的文本

分类

`curl 'http://localhost:5000/predict/?content=无知者无畏'`

跟B8相比优缺点

- 速度 - 数据转化成查询向量，并且缓存起来不用每次从资料库计算，应该比B8更快
- 注意 - 除了垃圾要学，非垃圾的也要学，若不学非垃圾数据，分类器会将所有都判断成垃圾
