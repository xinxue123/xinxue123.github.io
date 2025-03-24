机器学习算法
===========================

回归拟合
-----------------
利用torch实现一元线性回归,代码如下:

::

	import torch
	from torch import nn
	import matplotlib.pyplot as plt


	def get_fake_data(batch_size=10):
	    x = torch.rand(batch_size, 1) * 20
	    y = 2 * x + (1 + torch.randn(batch_size, 1)) * 3
	    return x, y


	if __name__ == '__main__':
	    criterion = nn.MSELoss() 
	    w = torch.rand(1, 1)  # 初始化权重
	    b = torch.rand(1, 1)  # 初始化截距
	    lr = 0.001  # 设置学习率 
	    plt.ion()
	    plt.figure(figsize=(8, 12))
	    for epoch in range(100):
	        plt.cla()
	        x, y = get_fake_data()
	        w = w.clone().detach().requires_grad_(True)
	        b = b.clone().detach().requires_grad_(True)
	        y_pre = x * w + b
	        loss = criterion(y_pre, y)
	        loss.backward()
	        # print(w.grad)
	        # print(b.grad)
	        w = w - lr * w.grad  # 更新权重
	        b = b - lr * b.grad  # 更新截距
	        print("weight is {}, intercept is {}".format(w, b))
	        plt.scatter(x.data, y.data)
	        plt.plot(x.data, y_pre.data)
	        plt.show()
	        plt.pause(1)

torch可视化
-------------------------
torch 可以采用visdom

1. 安装及启动

>>> pip3 install visdom
python -m visdom.server

2. 使用
::

	import visdom
	import numpy as np
	import time
	import torch

  	# 1. plot line
	vis = visdom.Visdom()
	x = np.arange(20)
	y = np.sin(x)
	vis.line(y, x, win="sin", opts={"title":"y=sinx(x)"})

	# 2. append point
	vis = visdom.Visdom()
	for i in range(100):
	    x = np.array([i])
	    y = np.sin(x)
	    vis.line(y, x, win="animal", update="append")
	    time.sleep(0.5)

EM算法
----------------------------------
EM 算法的手动实现

1. 首先计算E步,假设分布的均值和方差已知,分布的概率pi已知,此时可以计算每个样本在不同分布上的概率,计算公式如下：

 .. image:: gumma.png 
  :height: 171px
  :width:  447px
  :scale: 50 %
  :alt: alternate text
  :align: center

2. 计算M步,根据计算后的概率重新调整分布的均值和方差，以及pi,计算公式如下：

 .. image:: M_step.png 
  :height: 390px
  :width: 503 px
  :scale: 50 %
  :alt: alternate text
  :align: center

3. 具体实现代码如下
::

	import numpy as np
	from scipy.stats import multivariate_normal
	import matplotlib.pyplot as plt
	from sklearn.metrics.pairwise import pairwise_distances_argmin

	if __name__ == '__main__':
	    np.set_printoptions(suppress=True)
	    x1 = multivariate_normal([160, 55],[[18,12],[12,31]])  # 模拟一个高斯分布
	    x2 = multivariate_normal([173,65],[[22,28],[28,105]])
	    x1_data = x1.rvs(400)
	    x2_data = x2.rvs(400)
	    x = np.vstack((x1_data, x2_data))
	    u1 = np.min(x, axis=0)  # 初始化第一个分布的均值
	    u2 = np.max(x, axis=0)  # 初始化第二个分布的均值
	    uu1 = u1
	    uu2 = u2
	    var1 = np.diag(np.var(x, axis=0))  # 方差
	    var2 = np.diag(np.var(x, axis=0))  # 方差
	    pi = 0.5
	    for i in range(500):
	        n1 = multivariate_normal(mean=u1, cov=var1)  # 构造第一个分布
	        n2 = multivariate_normal(mean=u2, cov=var2)  # 构造第二个分布
	        n1_p = n1.pdf(x)
	        n2_p = n2.pdf(x)
	        gumma_1 = pi * n1_p
	        gumma_2 = (1-pi) * n2_p
	        g = pi * n1_p + (1-pi) * n2_p
	        gumma_1 = gumma_1 / g
	        gumma_2 = gumma_2 / g

	        # M step
	        u1 = np.dot(x.T, gumma_1) / np.sum(gumma_1)
	        u2 = np.dot(x.T, gumma_2) / np.sum(gumma_2)
	        var1 = np.dot((x - u1).T * gumma_1.T, x-u1) / np.sum(gumma_1)
	        var2 = np.dot((x - u2).T * gumma_2.T, x-u2) / np.sum(gumma_2)
	        pi = np.sum(gumma_1) / len(gumma_1)
	        print("第{}次：u1:".format(i), u1, " u2:", u2)

	    print(pi)
	    print(u1, u2)
	    print(var1)
	    print(var2)
	    plt.scatter(x1_data[:, 0], x1_data[:, 1])
	    plt.scatter(x2_data[:, 0], x2_data[:, 1])
	    plt.show()

HMM
-------------------------
隐马尔可夫的相关计算;

1. 前向算法公式

其中a是转移概率矩阵（隐状态转移到隐状态）,b是发射矩阵（隐状态转移到观测值）,pi是初始矩阵,P(O|lambda)是给定的以上三个值后看到观测值的概率大小

 .. image:: F_a.png 
  :height: 556px
  :width:  850px
  :scale: 30 %
  :alt: alternate text
  :align: center

2. 后向算法公式

 .. image:: B_a.png 
  :height: 562px
  :width:  799px
  :scale: 30 %
  :alt: alternate text
  :align: center

3. 前向概率与后向概率的关系

- 单个状态下的概率：

 .. image:: F_B.png 
  :height: 485px
  :width:  883px
  :scale: 30 %
  :alt: alternate text
  :align: center

- 两个状态下的联合概率：

 .. image:: M_F_B.png 
  :height: 559px
  :width:  1054px
  :scale: 30 %
  :alt: alternate text
  :align: center


实现脚本
::

	def calc_alpha(ob, a, b, pi):
	    """
	    前向算法： pi a b ob
	    """
	    alpha = np.zeros((pi.size, ob.size))
	    alpha[:, 0] = pi
	    alpha = (alpha.T * b[:, ob[0]]).T
	    for i in range(1, len(ob)):
	        for s in range(len(pi)):
	            alpha[s, i] = np.sum(alpha[:, i - 1] * a[:, s]) * b[s, ob[i]]

	    print(alpha)
	    return alpha


	def calc_beta(ob, a, b, pi):
	    """
	    后向算法: pi a b ob
	    """
	    beta = np.ones((pi.size, ob.size))
	    for i in range(ob.size - 2, -1, -1):
	        for s in range(pi.size):
	            beta[s, i] = np.sum(a[s, :] * b[:, ob[i + 1]] * beta[:, i + 1])
	    return beta
	    # beta[:,0] = beta[:,0] * b[:, ob[0]] * pi
	    # print(beta)
	    # print(np.sum(beta, axis=0))


	def bw(ob, a, b, pi):
		# 迭代求解 a b pi
	    alpha = calc_alpha(ob, a, b, pi)
	    beta = calc_beta(ob, a, b, pi)
	    pi = alpha[:, 0] * beta[:, 0]
	    pi = pi / np.sum(pi)

	    a_part = alpha[:, :-1] * beta[:, :-1]
	    a_2 = np.sum(a_part, axis=1)
	    a_1 = np.dot(alpha[:, :-1], (beta[:,1:] * b[:, ob[1:]]).T) * a
	    a = (a_1.T / a_2).T

	    b_1 = alpha * beta
	    b0 = np.sum(b_1[:, ob==0], axis=1)
	    b1 = np.sum(b_1[:, ob==1], axis=1)
	    b = np.c_[b0, b1]
	    b = (b.T / np.sum(b_1, axis=1)).T

	    return a, b, pi
	if __name__ == '__main__':
		# 简单测试数据
		pi = np.array([0.2, 0.4, 0.4])
		a = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
		b = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
		oo = np.array([0, 1, 0])
		a_1, b_1, pi_1 = bw(oo, a, b, pi)

最大熵思想实现分类
-----------------------------------
最大熵求解即求最大不确定性,本脚本采用torch训练模型，并使用模型进行预测。
::

	from sklearn.datasets import load_iris
	from sklearn.model_selection import train_test_split
	import pandas as pd
	import numpy as np
	import torch
	from torch.optim import Adam

	# 计算概率
	def cal_p(x_train, y_train, w, p_mean, p_std):
	    for l in range(p_std.shape[0]):
	        # print(type(x_train), type(y_train), type(w), type(p_std), type(p_mean))
	        res_std = p_std[l]
	        res_mean = p_mean[l]
	        res = torch.exp(-1/2 * w * ((x_train-res_mean)**2) / (res_std ** 2)) / np.sqrt(2*np.pi) / res_std
	    p = res[np.arange(len(res)),y_train.numpy()] / torch.cumsum(res, 1)[:,-1]
	    return p

	def predict(x_train, y_train, w, p_mean, p_std):
    	pp = torch.zeros((x_train.shape[0], p_std.shape[0]))
	    for l in range(len(p_std)):
	        # print(type(x_train), type(y_train), type(w), type(p_std), type(p_mean))
	        res_std = p_std[l]
	        res_mean = p_mean[l]
	        # tmp = torch.exp(-0.5 * w * ((x_train-res_mean)**2) / (res_std ** 2)) / np.sqrt(2*np.pi) / res_std
	        res = torch.sum(torch.exp(-0.5 * w * ((x_train-res_mean)**2) / (res_std ** 2)) / np.sqrt(2*np.pi) / res_std, 1)
	        pp[:,l] = res

	    # 概率归一化
	    for l in range(len(p_std)):
	        pp[:, l] = pp[:, l] / torch.sum(pp, 1)
	    return pp

	np.set_printoptions(precision=5,suppress=True)
	data = load_iris() # 加载鸢尾花数据集
	x = data.get("data")
	y = data.get("target")
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
	train_stats = pd.DataFrame(data=np.c_[x_train, y_train], columns=["1","2","3","4","class"])
	res_mean = train_stats.groupby("class").mean().reset_index() # 统计样本数据均值
	res_std = train_stats.groupby("class").std().reset_index() # 统计样本数据标准差
	res_std = torch.from_numpy(res_std.values[:,1:])
	res_mean = torch.from_numpy(res_mean.values[:,1:])
	w = torch.zeros(res_std.shape[1], dtype=torch.double, requires_grad=True)
	x_train = torch.from_numpy(x_train)
	y_train = torch.from_numpy(y_train)
	optimizer = Adam({w1:"weight"}, lr=0.1)

	# 训练模型
	for i in range(1000):
	    result = cal_p(x_train, y_train, w, p_mean=res_mean, p_std=res_std)
        result = torch.log(result)
	    loss = -torch.sum(result) # 最大似然

	    optimizer.zero_grad()
	    loss.backward()
	    optimizer.step()
	    print(w1)

	# 利用模型预测数据
	pp = predict(x_test, y_test, w, p_mean=res_mean, p_std=res_std)
	index = torch.argmax(pp, 1) # 选取概率最大的类别
	print(index)
	print(y_test)
	print(index - y_test)

基于Faster R-CNN的遥感影像飞机检测
----------------------------------------
本代码采用ResNet50作为backbone,并使用预训练参数，具体代码如下。
::
	import glob
	import os
	import numpy as np
	from osgeo import gdal_array
	from torch.utils.data import Dataset
	import visdom
	import cv2
	from torchvision.models.detection import FasterRCNN
	import torch
	from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
	from torchvision.models.utils import load_state_dict_from_url
	from torch import optim
	viz = visdom.Visdom()
	class Data(Dataset):
	    def __init__(self, path):
	        super(Data, self).__init__()
	        self.figs = glob.glob(os.path.join(path, "*.png"))
	    def __getitem__(self, item):
	        p = self.figs[item]
	        fig = gdal_array.LoadFile(p)[:3]
	        target = self.get_label(p)
	        return fig, target
	    def __len__(self):
        return len(self.figs)
    def get_label(self, p):
        # print(p)
        target = {}
        label = np.loadtxt(p.replace(".png",".txt"))
        label = label.reshape(-1, 13)
        label = np.array(label, dtype=np.int64)
        label = label[:, -4:]
        label[:, 2:] = label[:, 2:] + label[:, :2]
        # label = np.array(label, dtype=int)
        l = torch.from_numpy(label)
        target["boxes"] = l
        target["labels"] = torch.ones(label.shape[0], dtype=torch.int64)
        return target
	def get_batch(data, batch=5):
	    idx = np.random.randint(0, len(data), batch)
	    images = []
	    targets = []
	    for i in idx:
	        x, y = data[i]
	        # x = x[np.newaxis, :]
	        images.append(torch.from_numpy(x).float()/255)
	        targets.append(y)
	    return images, targets
	def draw_rectangle(img, rectangles):
	    img = img.detach().numpy().copy()
	    img = np.array(img*255, dtype=np.uint8).copy()
	    rectangles = rectangles.detach().numpy()
	    rectangles = np.array(rectangles, dtype=np.int32)
	    img = img.transpose(1, 2, 0)
	    for i in range(rectangles.shape[0]):
	        img[:,:,0] = cv2.rectangle(img[:,:, 0], (int(rectangles[i][0]), int(rectangles[i][1])), (int(rectangles[i][2]), int(rectangles[i][3])), 255,thickness=2 )
	    img = img.transpose(2, 0, 1)
	    return img/255
	def load_model():
	    backbone = resnet_fpn_backbone('resnet50', True)
	    state_dict = load_state_dict_from_url("https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",progress=True)
	    keys = []
	    for k, v in state_dict.items():
	        if k.startswith("backbone"):
	            pass
	        else:
	            keys.append(k)
	    for k in keys:
	        state_dict.pop(k)
	    model = FasterRCNN(backbone, 2)
	    model.load_state_dict(state_dict, strict=False)
	    for name,p in model.named_parameters():
	        if name.startswith("backbone"):
	            p.requires_grad_(False)
	    para = [p for p in model.parameters() if p.requires_grad]
	    # model.transform = None
	    return model, para
	data = Data(r"G:\目标检测\中科院大学高清航拍目标数据集合\PLANE")
	model, para = load_model()
	optimizer = optim.Adam(para, lr=1e-3)
	# model = torch.load("detect_model")
	for i in range(1000000):
	    model = model.train()
	    x, y = get_batch(data, 1)
	    loss = model(x, y)
	    t_loss = torch.tensor(0.0)
	    for k, v in loss.items():
	        t_loss += v
	    optimizer.zero_grad()
	    t_loss.backward()
	    optimizer.step()
	    print(t_loss)
	    if i%100==0:
	        torch.save(model, "detect_model")
	    model = model.eval()
	    pred = model(x)
	    img = torch.squeeze(x[0])
	    img = draw_rectangle(img, pred[0]["boxes"][pred[0]["scores"]>0.8])
	    viz.image(img, "image")

效果图如下:

 .. image:: d1.png 
  :height: 595px
  :width:  1147px
  :scale: 40 %
  :alt: alternate text
  :align: center


 .. image:: d2.png 
  :height: 595px
  :width:  1147px
  :scale: 40 %
  :alt: alternate text
  :align: center


基于DeepLabv03的遥感影像道路提取
----------------------------------------
本代码采用ResNet50作为backbone,并使用预训练参数，具体代码如下。
::
	model = deeplabv3_resnet50(pretrained=True)
	aux_classifier = nn.Sequential(
	    nn.Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
	    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
	    nn.ReLU(),
	    nn.Dropout(p=0.1, inplace=False),
	    nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
	)
	print(model)
	# for i in model.parameters():
	#     i.requires_grad_(False)
	model.aux_classifier = aux_classifier


	class Data(Dataset):
	    def __init__(self, block=1024):
	        super(Data, self).__init__()
	        self.block = block
	        img_path = r"E:\图像分割\road\images"
	        self.label_path = r"E:\图像分割\road\gt"
	        self.files = glob.glob(os.path.join(img_path, "*.tiff"))

	    def __getitem__(self, item):
	        path = self.files[item]
	        train_x = np.array(gdal_array.LoadFile(path), np.float32)
	        path = os.path.join(self.label_path, os.path.basename(path))
	        path = path.replace(".tiff", ".tif")
	        mask = np.all(train_x == 255, axis=0)
	        train_y = gdal_array.LoadFile(path)
	        train_y[mask] = 0
	        train_y[train_y == 255] = 1

	        size = train_x.shape
	        y_end = size[1] - self.block - 1
	        x_end = size[2] - self.block - 1
	        y_rand = np.random.randint(y_end)
	        x_rand = np.random.randint(x_end)
	        train_x = train_x[:, y_rand:y_rand + self.block, x_rand:x_rand + self.block]
	        train_y = train_y[y_rand:y_rand + self.block, x_rand:x_rand + self.block]
	        return train_x, train_y

	    def __len__(self):
	        return len(self.files)


	model = torch.load("seg_model")
	criterion = nn.CrossEntropyLoss()
	para = []
	for i in model.parameters():
	    if i.requires_grad:
	        para.append(i)
	optimizer = optim.Adam(para, betas=(0.5, 0.99))

	viz = visdom.Visdom()

	if __name__ == '__main__':
	    data = Data()
	    data_iter = DataLoader(data, 2)
	    for i in range(100000):
	        for x, y in data_iter:
	            y = y.long()
	            pre = model(x)["aux"]
	            loss = criterion(pre, y)
	            print(loss)
	            optimizer.zero_grad()
	            loss.backward()
	            optimizer.step()
	            viz.image(x[0], win="img1")
	            viz.image(torch.argmax(pre, 1)[0].float(), win="img2")
	            viz.image(y[0].float(), win="img3")
	            print("save model")
	            # torch.save(model, "seg_model")
	    print("finish!")


效果图如下:

 .. image:: seg01.png 
  :height: 425px
  :width:  1294px
  :scale: 40 %
  :alt: alternate text
  :align: center


 .. image:: seg02.png 
  :height: 425px
  :width:  1294px
  :scale: 40 %
  :alt: alternate text
  :align: center
