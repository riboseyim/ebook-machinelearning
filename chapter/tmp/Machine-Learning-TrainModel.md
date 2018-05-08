---
title: Machine Learning:Training Models
date: 2018-02-27 14:23:57
tags: [Machine-Learning,数学与算法,Developer,DevOps,架构师]
---
## 摘要

<!--more-->

## Previous

前文回顾：
- [Machine Learning(一):基于 TensorFlow 实现宠物血统智能识别](https://riboseyim.github.io/2018/01/17/Machine-Learning-TensorFlow/)
- [Machine Learning (二) : 宠物智能识别之 Using OpenCV with Node.js](https://riboseyim.github.io/2018/01/15/Machine-Learning-OpenCV/)

上面的文章中提到了机器学习的预测结果受模型质量的影响很大，如果想要取得好的效果需要通过训练增强优化。


#### Training Data
```bash
curl http://download.tensorflow.org/example_images/flower_photos.tgz \| tar xz -C tf_files
```

#### Training the Network

```bash
git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
```

```python
python scripts/retrain.py
--image_dir=tf_files/flower_photos
--output_graph=tf_files/retrained_graph.pb
--output_labels=tf_files/retrained_labels.txt
```

#### Test:Using the Retrained Model

```python
python scripts/label_image.py --image data/daisy.jpg
```

## Model
```js
node {
  name: "DecodeJpeg/contents"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "\377\330......."
}
}
}
}

node {
name: "softmax/biases"
op: "Const"
attr {
key: "dtype"
value {
type: DT_FLOAT
}
}
attr {
key: "value"
value {
tensor {
  dtype: DT_FLOAT
  tensor_shape {
    dim {
      size: 1008
    }
  }
  tensor_content: "\312\277\006"
}
}
}
}
node {
name: "softmax/logits/MatMul"
op: "MatMul"
input: "pool_3/_reshape"
input: "softmax/weights"
attr {
key: "T"
value {
type: DT_FLOAT
}
}
attr {
key: "transpose_a"
value {
b: false
}
}
attr {
key: "transpose_b"
value {
b: false
}
}
}
node {
name: "softmax/logits"
op: "BiasAdd"
input: "softmax/logits/MatMul"
input: "softmax/biases"
attr {
key: "T"
value {
type: DT_FLOAT
}
}
}
node {
name: "softmax"
op: "Softmax"
input: "softmax/logits"
attr {
key: "T"
value {
type: DT_FLOAT
}
}
}
```

#### Optional Parameters

#### Questions

```
22:17:28.523085: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
22:17:28 NodeDef mentions attr 'dilations' not in Op<name=Conv2D; signature=input:T, filter:T -> output:T; attr=T:type,allowed=[DT_HALF, DT_FLOAT]; attr=strides:list(int); attr=use_cudnn_on_gpu:bool,default=true; attr=padding:string,allowed=["SAME", "VALID"]; attr=data_format:string,default="NHWC",allowed=["NHWC", "NCHW"]>; NodeDef: conv/Conv2D = Conv2D[T=DT_FLOAT, data_format="NHWC", dilations=[1, 1, 1, 1], padding="VALID", strides=[1, 2, 2, 1], use_cudnn_on_gpu=true](Mul, conv/conv2d_params). (Check whether your GraphDef-interpreting binary is up to date with your GraphDef-generating binary.).

22:22:02.629987: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.2 AVX AVX2 FMA
```

## About Data Sets
- [Fun with small image data-sets (Part 2)](https://towardsdatascience.com/fun-with-small-image-data-sets-part-2-54d683ca8c96)

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


## 参考文献
- [综述论文：机器学习中的模型评价、模型选择与算法选择 | 2018-02-02 机器之心](https://mp.weixin.qq.com/s/J75ZdrNCSwBO1y9o84THQg)
- [Plug & Play Machine Learning Models in GoLang |  BAYESIAN CLASSIFICATION](https://dev.to/michaeljtaylor0/plug--play-machine-learning-models-in-golang--fc0)
- [Train Your Machine Learning Models on Google’s GPUs for Free — Forever](https://hackernoon.com/train-your-machine-learning-models-on-googles-gpus-for-free-forever-a41bd309d6ad)
- [How do we ‘train’ neural networks ?](https://towardsdatascience.com/how-do-we-train-neural-networks-edd985562b73)
