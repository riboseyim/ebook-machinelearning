---
title: Machine Learning(二):宠物智能识别之 Using OpenCV with Node.js
date: 2018-01-15 17:01:06
tags: [Nodejs,Developer,Machine-Learning]
---
## 摘要
- 计算机视觉库：OpenCV
- Using OpenCV with Node.js

<!--more-->

## Privious

OpenCV ( Open Source Computer Vision Library )，是一个基于C/C++语言的计算机视觉库，在跨平台图像/视频处理、模式识别、人机交互、机器人等领域有广泛的应用。

OpenCV 包含机器学习库，支持传统的机器学习算法（决策树、朴素贝叶斯、支持向量机、随机森林等），近期的版本演进侧重于增强深度学习的支持，例如 [OpenCV 3.3](https://opencv.org/opencv-3-3.html) 将深度神经网络（Deep neural networks，DNN ) 提升到主代码库（promote DNN module from opencv_contrib to the main repository），2017 年 12 月发布的 OpenCV 3.4 针对 R-CNN 进行了性能优化。

OpenCV 的主要编程是 C++ ，大部分的接口也是基于 C++，但它仍然保留着很多 C 接口（功能不完整）。绑定语言（binding）有 Python，java 和 MATLAB /OCTAVE ，另外还有一些其他语言的包装器（wrapper）如 C# , Perl , Haskell 和 Ruby。**[opencv4nodejs](https://github.com/justadudewhohacks/opencv4nodejs)** 项目是一个支持所有 OpenCV 3 的 Node.js 绑定，有助于弥补 JavaScript 缺乏计算机视觉实现的不足，为 Node.js 优势应用场景的选型中提供了更多选择（例如使用 WebSocket 推送技术创建实时 Web 应用）。

前文回顾：[Machine Learning(一):基于 TensorFlow 实现宠物血统智能识别](https://riboseyim.github.io/2018/01/17/Machine-Learning-TensorFlow/) 中演示了一个图像识别的案例，我们来看看基于 OpenCV + Node.js 是如何实现的：

## Using OpenCV with Node.js

![](http://omb2onfvy.bkt.clouddn.com/ML-Tensorflow-Theme-Dog.png)

#### Enviroment

```bash
$ cmake --version
cmake version 3.10.2

$ brew install cmake

$ brew install opencv3
# dependencies for opencv: eigen, lame, x264, xvid, ffmpeg, libpng,
# libtiff, ilmbase, openexr, gdbm, python, xz, python3, numpy, tbb

$ mkdir project-opencv-demo
$ cd project-opencv-demo
$ npm init
$ npm install --save opencv4nodejs
```

#### Load InceptionModel

Tensorflow Inception Model 是一个已经被训练好的模型，可以识别数千类对象，只要将图像输入就可以输出推测的一个分类概率。Tensorflow Inception Model 包括 ‘graph.pb’ 和 ‘label_strings.txt’ 两个文件，使用之前需要先加载。

```js
const cv = require('opencv4nodejs');
//const cv = require('../');
const fs = require('fs');
const path = require('path');

if (!cv.xmodules.dnn) {
  throw new Error('exiting: opencv4nodejs compiled without dnn module');
}

// replace with path where you unzipped inception model
const inceptionModelPath = './models/tf-inception'

const modelFile = path.resolve(inceptionModelPath, 'tensorflow_inception_graph.pb');
const classNamesFile = path.resolve(inceptionModelPath, 'imagenet_comp_graph_label_strings.txt');
if (!fs.existsSync(modelFile) || !fs.existsSync(classNamesFile)) {
  console.log('exiting: could not find inception model');
  console.log('download the model from: https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip');
  return;
}
console.log('load models:'+inceptionModelPath)

// read classNames and store them in an array
const classNames = fs.readFileSync(classNamesFile).toString().split("\n");

// initialize tensorflow inception model from modelFile
const net = cv.readNetFromTensorflow(modelFile);
```

#### Image Classify

读取图片存储为 Blob 格式，调用 **net.forward()** （图像作为输入参数），此处我们仅输出概率高于 5% 的分类。

```js
const classifyImg = (img) => {
  // inception model works with 224 x 224 images, so we resize
  // our input images and pad the image with white pixels to
  // make the images have the same width and height
  const maxImgDim = 224;
  const white = new cv.Vec(255, 255, 255);
  const imgResized = img.resizeToMax(maxImgDim).padToSquare(white);

  // network accepts blobs as input
  const inputBlob = cv.blobFromImage(imgResized);
  net.setInput(inputBlob);

  // forward pass input through entire network, will return
  // classification result as 1xN Mat with confidences of each class
  const outputBlob = net.forward();

  // find all labels with a minimum confidence
  const minConfidence = 0.05;
  const locations =
    outputBlob
      .threshold(minConfidence, 1, cv.THRESH_BINARY)
      .convertTo(cv.CV_8U)
      .findNonZero();

  const result =
    locations.map(pt => ({
      confidence: parseInt(outputBlob.at(0, pt.x) * 100) / 100,
      className: classNames[pt.x]
    }))
      // sort result by confidence
      .sort((r0, r1) => r1.confidence - r0.confidence)
      .map(res => `${res.className} (${res.confidence})`);

  return result;
}
```

#### Test

```js
const testData = [
  {
    image: './data/IMG_3560.png',
    label: 'Yan Dog'
  },
  {
    image: './data/IMG_3608.png',
    label: 'Yang Dog'
  }
];

testData.forEach((data) => {
  const img = cv.imread(data.image);
  console.log('%s,%s: ', data.image,data.label);

  const predictions = classifyImg(img);
  predictions.forEach(p => console.log(p));

  //cv.imshowWait('img', img);

  console.log("---------finish---------");
});
```

![IMG_3608.png](http://omb2onfvy.bkt.clouddn.com/Tensorflow-Demo-Dog-2.png)

```
$ npm run tf-classify

> node ./tf-classify.js
load models:./models/tf-inception

-------------------------------
./data/IMG_3560.png,Yan Dog:
[ INFO:0] Initialize OpenCL runtime...
潘布魯克威尔斯柯基犬 Pembroke (0.83)
-------------------------------
./data/IMG_4423.png,Yang Dog:
吉娃娃 Chihuahua (0.89)
Pembroke (0.07)
-------------------------------
./data/IMG_3608.png,Yang Dog:
玩具梗 toy terrier (0.22)
美国斯塔福德郡梗 American Staffordshire terrier (0.2)
吉娃娃 Chihuahua (0.14)
斯塔福德郡牛头梗 Staffordshire bullterrier (0.12)
比特犬 whippet (0.05)
-------------------------------
```

问题：对比之前 [Machine Learning(一):基于 TensorFlow 实现宠物血统智能识别](https://riboseyim.github.io/2018/01/17/Machine-Learning-TensorFlow/) 的预测值，两次识别的结果很接近，但是又有不同，这是为什么呢？请注意后续更新。

## OpenCV 概要

OpenCV ( Open Source Computer Vision Library )，是一个基于C/C++语言的跨平台图像/视频处理函数库。OpenCV 由英特尔公司于1999年发起并参与开发，以 BSD 许可证授权发行，可以在商业和研究领域中免费使用。OpenCV 主要用于开发实时的图像处理、计算机视觉以及模式识别程序。

OpenCV 的主要编程是 C++ ，大部分的接口也是基于 C++，但它仍然保留着很多 C 接口（不完整）。绑定语言（binding）有 Python，java 和 MATLAB /OCTAVE ，另外还有一些其他语言的包装器（wrapper）如 C# , Perl , Haskell 和 Ruby。

#### OpenCV 版本

OpenCV 第一个预览版本于 2000 年在 IEEE Conference 公开，目前每6个月就会有一个官方版本，由一个商业公司赞助的独立小组开发。
OpenCV 1.0：2006年发布
OpenCV 2.0：2009年10月发布，主要更新包括 C++ 接口
OpenCV 2.3：2011年6月发布，主要更新包括移动终端兼容性（NDK-Build）
OpenCV 3.0：2015年6月发布
OpenCV 3.3: 2017年8月发布，主要更新包括深度学习（promote DNN module from opencv_contrib to the main repository）
OpenCV 3.4: 2017年12月发布，主要更新包括 DNN 模块改进（包括 R-CNN 性能优化), Javascript 绑定和 OpenCL 实现

```bash
#查看版本
$ pkg-config --modversion opencv
3.4.0
```

#### OpenCV 主要模块
- cv 核心函数库
- cvaux 辅助函数库
- cxcore 数据结构与线性代数库
- highgui GUI 函数库，包括用户界面、读/写图像及视频
- ml 机器学习函数库，包括统计模型、贝叶斯、最近邻居、支持向量机、决策树、随机树、最大期望、神经网络等，详见[Machine Learning:机器学习算法](https://riboseyim.github.io/2018/02/10/Machine-Learning-Algorithms/)。
- gpu GPU加速,GPU模块及数据结构，包含图像处理与分析模块

#### OpenCV 主要功能
- 图像数据操作（内存分配与释放 allocation & release，图像复制 copying、设定和转换 setting & conversion）
- 矩阵/向量数据操作及线性代数运算（矩阵乘积、矩阵方程求解、特征值、奇异值分解）
- 支持多种动态数据结构（链表、队列、数据集、树、图）
- 基本图像处理（去噪、边缘检测、角点检测、采样与插值、色彩变换、形态学处理、直方图、图像金字塔结构）
- 结构分析（连通域/分支、轮廓处理、距离转换、图像矩、模板匹配、霍夫变换、多项式逼近、曲线拟合、椭圆拟合、狄劳尼三角化）
- 图像/视频的输入输出（支持文件或摄像头的输入，图像/视频文件的输出）
- 摄像头定标（寻找和跟踪定标模式、参数定标、基本矩阵估计、单应矩阵估计、立体视觉匹配）
- 运动分析（光流 optical flow、动作分割 motion segmentation、目标跟踪 tracking）

#### OpenCV 基本数据类型

- CvPoint:表示一个坐标为整数的二维点
- CvSize:表示矩阵框大小，以像素为精度。
- CvRect:通过方形左上角坐标和方形的高和宽来确定一个矩形区域
- CvScalar:用来存放像素值（ double 数组，不一定是灰度值）

```C
typedef  struct  CvPoint
{
    int x;//图像中点的x坐标
    int y;//图像中点的y坐标
}

typedef struct CvSize
{
    int width; //矩形宽
    int height; //矩形高
}

typedef struct CvRect 　　
{ 　　
    int x; //方形的左上角的x-坐标 　　
    int y; //方形的左上角的y-坐标　
    int width; //宽 　　
    int height; //高
} 　

typedef struct CvScalar
{
    double val[4];
}
```

#### OpenCV 与机器学习

OpenCV 包含机器学习库，支持以下算法：

- Boosting
- Decision tree learning
- Gradient boosting trees
- Expectation-maximization algorithm
- k-nearest neighbor algorithm
- Naive Bayes classifier
- Artificial neural networks
- Random forest
- Support vector machine (SVM)
- Deep neural networks (DNN) （[OpenCV 3.3](https://opencv.org/opencv-3-3.html) promote DNN module from opencv_contrib to the main repository）

![](http://omb2onfvy.bkt.clouddn.com/ML-Algorithm-OpenCV.png)

#### OpenCV 资源
- [OpenCV github repo](https://github.com/opencv/opencv)
- [Using OpenCV with Node.js](https://hub.docker.com/r/justadudewhohacks/opencv-nodejs/)
- [opencv4nodejs](https://github.com/justadudewhohacks/opencv4nodejs)
- [OpenCV编程简介 Introduction to programming with OpenCV](http://wiki.opencv.org.cn/index.php/OpenCV_%E7%BC%96%E7%A8%8B%E7%AE%80%E4%BB%8B%EF%BC%88%E7%9F%A9%E9%98%B5/%E5%9B%BE%E5%83%8F/%E8%A7%86%E9%A2%91%E7%9A%84%E5%9F%BA%E6%9C%AC%E8%AF%BB%E5%86%99%E6%93%8D%E4%BD%9C%EF%BC%89#.EF.BC.884.EF.BC.89____.E8.AE.BE.E7.BD.AE.2F.E8.8E.B7.E5.8F.96.E6.84.9F.E5.85.B4.E8.B6.A3.E5.8C.BA.E5.9F.9FROI:)

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
- [The 5 Computer Vision Techniques That Will Change How You See The World](https://heartbeat.fritz.ai/the-5-computer-vision-techniques-that-will-change-how-you-see-the-world-1ee19334354b)
- **Paper** [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)
- [Node.js meets OpenCV’s Deep Neural Networks — Fun with Tensorflow and Caffe](https://medium.com/@muehler.v/node-js-meets-opencvs-deep-neural-networks-fun-with-tensorflow-and-caffe-ff8d52a0f072)
- [Build an Image Recognition API with Go and TensorFlow](https://outcrawl.com/image-recognition-api-go-tensorflow/)
- [Train your own image classifier with Inception in TensorFlow | Wednesday, March 09, 2016 | Google Research Blog](https://research.googleblog.com/2016/03/train-your-own-image-classifier-with.html)
- [Node.js + OpenCV for Face Recognition](https://medium.com/@muehler.v/node-js-opencv-for-face-recognition-37fa7cb860e8)
- [Node.js + face-recognition.js : Simple and Robust Face Recognition using Deep Learning](https://medium.com/@muehler.v/node-js-face-recognition-js-simple-and-robust-face-recognition-using-deep-learning-ea5ba8e852)
- [Node.js meets OpenCV’s Deep Neural Networks — Fun with Tensorflow and Caffe](https://medium.com/@muehler.v/node-js-meets-opencvs-deep-neural-networks-fun-with-tensorflow-and-caffe-ff8d52a0f072)
- [Machine Learning is Fun! Part 4: Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)
- [Machine Learning with OpenCV and JavaScript: Recognizing Handwritten Letters using HOG and SVM](https://medium.com/@muehler.v/machine-learning-with-opencv-and-javascript-part-1-recognizing-handwritten-letters-using-hog-and-88719b70efaa)
- [Live Face Detection in Go using OpenCV and MachineBox | video](https://www.youtube.com/watch?v=rbZeZNVA-Q4)
- [雅虎开源色情图片检测神经网络](http://www.infoq.com/cn/news/2016/10/YAHOO-pornographic-detection-ne?utm_campaign=infoq_content&utm_source=infoq&utm_medium=feed&utm_term=global)
