---
title: Machine Learning:WordBook
date: 2018-03-21 11:08:01
tags: [Machine-Learning]
---
## 摘要

<!--more-->
## Basic

## 数学概念

- 泰勒级数 Taylor series

用无限项连加式——级数来表示一个函数，这些相加的项由函数在某一点的导数求得，以 1715 年发表了泰勒公式的英国数学家布鲁克·泰勒（Sir Brook Taylor）来命名。通过函数在自变量零点的导数求得的泰勒级数又叫做麦克劳林级数，以苏格兰数学家科林·麦克劳林的名字命名。

- 导数 Derivative

导数（Derivative）是微积分学中重要的基础概念。一个函数在某一点的导数描述了这个函数在这一点附近的变化率。导数的本质是通过极限的概念对函数进行局部的线性逼近。

- 偏导数 Partial Derivatives

在数学中，一个多变量的函数的偏导数是它关于其中一个变量的导数，而保持其他变量恒定（相对于全导数，在其中所有变量都允许变化）

- 超参数 Hyper Parameter

在机器学习中，超参数是在学习过程开始之前设置的一个参数。相比之下，其他参数的值是通过训练得到的。

不同的模型训练算法需要不同的超参数，一些简单的算法（如普通最小二乘回归）不需要。给定这些超参数，训练算法从数据中学习参数。例如，LASSO 是一种算法，将正则化超参数添加到普通最小二乘回归中，它必须在通过训练算法估计参数之前设置。在深度学习中，常见的超参数有：学习速率，迭代次数，层数，每层神经元的个数等。

- 激活函数 Activation Function

在 计算网络中， 一个节点的激活函数定义了该节点在给定的输入或输入的集合下的输出

## Deep Learning

#### Neural Network
- 神经网络（Neural Network，缩写 NN）
- 人工神经网络（Artificial Neural Network，缩写 ANN）

人工神经网络（Artificial neural network，缩写ANN），简称神经网络（neural network，缩写NN）或类神经网络，是一种模仿生物神经网络(动物的中枢神经系统，特别是大脑)的结构和功能的数学模型或计算模型。神经网络由大量的人工神经元联结进行计算。大多数情况下人工神经网络能在外界信息的基础上改变内部结构，是一种自适应系统。现代神经网络是一种非线性统计性数据建模工具，常用来对输入和输出间复杂的关系进行建模，或用来探索数据的模式。

#### Convolutional Neural Network
- 卷积神经网络(Convolutional Neural Network,缩写 CNN )
In machine learning, a convolutional neural network (CNN, or ConvNet) is a type of feed-forward artificial neural network in which the connectivity pattern between its neurons is inspired by the organization of the animal visual cortex, whose individual neurons are arranged in such a way that they respond to overlapping regions tiling the visual field. Convolutional networks were inspired by biological processes and are variations of multilayer perceptrons designed to use minimal amounts of preprocessing. They have wide applications in image and video recognition, recommender systems and natural language processing.

#### RNN
- RNN 一般指代时间递归神经网络(Recurrent Neural Network)
- 结构递归神经网络(Recursive Neural Network)

A recurrent neural network (RNN) is a class of artificial neural network where connections between units form a directed cycle. This creates an internal state of the network which allows it to exhibit dynamic temporal behavior. Unlike feedforward neural networks, RNNs can use their internal memory to process arbitrary sequences of inputs. This makes them applicable to tasks such as unsegmented connected handwriting recognition or speech recognition.

递归神经网络（RNN）是两种人工神经网络的总称。时间递归神经网络的神经元间连接构成有向图，而结构递归神经网络利用相似的神经网络结构递归构造更为复杂的深度网络。单纯递归神经网络因为无法处理随着递归，权重指数级爆炸或消失的问题（Vanishing gradient problem），难以捕捉长期时间关联；而结合不同的LSTM可以很好解决这个问题。

#### GAN
Generative Adversarial Network，生成对抗网络
非监督式学习的一种方法，通过让两个神经网络相互博弈的方式进行学习。该方法由伊恩·古德费洛等人于2014年提出）

#### Autoencoders 自动编码

#### Reinforcement Learning 强化学习
Reinforcement learning is an area of machine learning inspired by behaviorist psychology, concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward. The problem, due to its generality, is studied in many other disciplines, such as game theory, control theory, operations research, information theory, simulation-based optimization, multi-agent systems, swarm intelligence, statistics, and genetic algorithms. In the operations research and control literature, the field where reinforcement learning methods are studied is called approximate dynamic programming. The problem has been studied in the theory of optimal control, though most studies are concerned with the existence of optimal solutions and their characterization, and not with the learning or approximation aspects. In economics and game theory, reinforcement learning may be used to explain how equilibrium may arise under bounded rationality.

强化学习是机器学习中的一个领域，强调如何基于环境而行动，以取得最大化的预期利益。其灵感来源于心理学中的行为主义理论，即有机体如何在环境给予的奖励或惩罚的刺激下，逐步形成对刺激的预期，产生能获得最大利益的习惯性行为。这个方法具有普适性，因此在其他许多领域都有研究，例如博弈论、控制论、运筹学、信息论、模拟优化方法、多主体系统学习、群体智能、统计学以及遗传算法。在运筹学和控制理论研究的语境下，强化学习被称作“近似动态规划”（approximate dynamic programming，ADP）。在最优控制理论中也有研究这个问题，虽然大部分的研究是关于最优解的存在和特性，并非是学习或者近似方面。在经济学和博弈论中，强化学习被用来解释在有限理性的条件下如何出现平衡。

## Open Source Library

#### TensorFlow
TensorFlow is an open source software library for machine learning in various kinds of perceptual and language understanding tasks. It is a second-generation API which is currently used for both research and production by 50 different teams in dozens of commercial Google products, such as speech recognition, Gmail, Google Photos, and Search. These teams had previously used DistBelief, a first-generation API. TensorFlow was originally developed by the Google Brain team for Google's research and production purposes and later released under the Apache 2.0 open source license on November 9, 2015.

#### Scikit-Learn 机器学习算法库
Scikit-Learn(formerly scikits.learn) is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.

#### COCO SSD
Ok that worked pretty well, but how do we deal with images that show multiple objects. Well to recognize multiple objects in a single image, we will utilize what’s called a Single Shot Multibox Detector (SSD). In our second example we will look at a SSD model trained with the COCO (Common Object in Context) dataset. The model we are using has been trained on 84 different classes.

## 扩展阅读:[《The Machine Learning Master》](https://www.gitbook.com/book/riboseyim/machine-learning)
![](http://p11slcnom.bkt.clouddn.com/banner-MLM-201803.png)
- [Machine Learning(一):基于 TensorFlow 实现宠物血统智能识别](https://riboseyim.github.io/2018/01/17/Machine-Learning-TensorFlow/)
- [Machine Learning(二):宠物智能识别之 Using OpenCV with Node.js](https://riboseyim.github.io/2018/01/15/Machine-Learning-OpenCV/)
- [Machine Learning:机器学习项目](https://riboseyim.github.io/2018/02/09/Machine-Learning-Projects/)
- [Machine Learning:机器学习算法](https://riboseyim.github.io/2018/02/10/Machine-Learning-Algorithms/)
- [Machine Learning:如何选择机器学习算法](https://riboseyim.github.io/2018/04/02/Machine-Learning-Algorithms-Sheet/)
- [Machine Learning:神经网络基础](https://riboseyim.github.io/2018/05/07/Machine-Learning-Neural-Network)
- [Machine Learning:机器学习书单](https://riboseyim.github.io/2018/01/25/Machine-Learning-Books/)
- [Machine Learning:人工智能媒体报道集](https://riboseyim.github.io/2017/08/29/Machine-Learning-News)
- [Machine Learning:机器学习技术与知识产权法](https://riboseyim.github.io/2018/02/16/Machine-Learning-Law/)
- [Machine Learning:经济学家谈人工智能](https://riboseyim.github.io/2018/03/09/Machine-Learning-Economist/)
- [数据可视化（三）基于 Graphviz 实现程序化绘图](https://riboseyim.github.io/2017/09/15/Visualization-Graphviz/)
