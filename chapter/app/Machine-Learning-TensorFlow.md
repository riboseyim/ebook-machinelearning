# 基于 TensorFlow 实现机器学习应用开发

- Machine Learning Workflow
- Problem: 宠物分类、勋章识别、美女打分
- Demo: Hello TensorFlow !
- TensorFlow C library / Go binding

## Machine Learning Workflow
- Define the problem. What problems do you want to solve?
- Start simple. Be familiar with the data and the baseline results.
- Then try something more complicated.

## Problem

人类喜欢将所有事物都纳入鄙视链的范畴，宠物当然也不例外。一般来说，拥有一只纯种宠物可以让主人占据鄙视链的云端，进而鄙视那些混血或者流浪宠物。甚至还发展出了专业的鉴定机构，可以颁发《血统证明书》。但是考究各类纯种鉴定的常规方法，主要标准是眼睛的大小、颜色、鼻子的特点、身躯长度、尾巴特征、毛发等特征，当然也包括一些比较玄幻的属性，例如宠物家族的个性、气质等等。

[外军研究：美军授勋及嘉奖制度观察](https://riboseyim.github.io/2017/04/09/Medal/)一文中提到，世界各国军队都有自己的制服、军衔、勋章体系，它们既是军人荣誉的体现，也包含了丰富的职业信息。但是体系过于庞大也会带来识别困难，例如下图中的两位美军士兵，是否可以有一种方案可以自动、准确地识别各类徽章的意义呢？
![](http://riboseyim-qiniu.riboseyim.com/USArmy-Medal-Navy-Seg-Notes-1.png)

中文网络上有一个特殊名词：颜值。通常用来表示人物颜容英俊或靓丽的数值。人们希望有一个衡量标准可以用来评价、测量和比较人物容貌，许多社交软件甚至可以利用计算机视觉识别技术分析颜值、年龄、性别，甚至与好友一起进行颜值 PK ，当然这些软件的 “颜值” 算法总是备受争议。
- [哪种类型算是中国式标准美女？](https://www.zhihu.com/question/56607562)

其实以上三种场景本质上都是图像识别，可以概括为一种基于外观的分类（或者说“打分”）需求，接下来我试图基于机器学习的方法来解决这些问题。

## Demo: Hello TensorFlow !

>Tensorflow is not a Machine Learning specific library, instead, is a general purpose computation library that represents computations with graphs.

TensorFlow 开源软件库（Apache 2.0 许可证），最初由 Google Brain 团队开发。TensorFlow 提供了一系列算法模型和编程接口，让我们可以快速构建一个基于机器学习的智能服务。对于开发者来说，目前有四种编程接口可供选择：

- C++ source code: Tensorflow 核心基于 C++ 编写，支持从高到低各个层级的操作;
- Python bindings & Python library: 对标 C++ 实现，支持 Python 调用 C++ 函数;
- Java bindings;
- Go binding;

下面是一个简单的实例：

![](http://riboseyim-qiniu.riboseyim.com/Tensorflow-Inception-Go-C.png)

#### 环境准备

- 安装 TensorFlow  C library,包含一个头文件 c_api.h 和 libtensorflow.so
```bash
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.5.0.tar.gz

## options
TF_TYPE="cpu" # Change to "gpu" for GPU support
TF_VERSION='1.5.0'
curl -L \
  "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-$(go env GOOS)-x86_64-${TF_VERSION}.tar.gz" |
```

- 安装 Go 语言环境,参考：[玩转编程语言：Golang](https://riboseyim.github.io/2017/05/05/Language-Go-lang/)

- 安装 Tensorflow Go binding library
```
go get github.com/tensorflow/tensorflow/tensorflow/go
go get github.com/tensorflow/tensorflow/tensorflow/go/op
```

- 下载模型（demo model)，包含一个标签文件 label_strings.txt 和 graph.pb
```bash
mkdir model
wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip -O model/inception.zip
unzip model/inception.zip -d model
chmod -R 777 model
```

#### Tensorflow Model Function

```go
//Loading TensorFlow model
func loadModel() error {
  // Load inception model
  model, err := ioutil.ReadFile("./model/tensorflow_inception_graph.pb")
  if err != nil {
    return err
  }
  graph = tf.NewGraph()
  if err := graph.Import(model, ""); err != nil {
    return err
  }
  // Load labels
  labelsFile, err := os.Open("./model/imagenet_comp_graph_label_strings.txt")
  if err != nil {
    return err
  }
  defer labelsFile.Close()
  scanner := bufio.NewScanner(labelsFile)
  // Labels are separated by newlines
  for scanner.Scan() {
    labels = append(labels, scanner.Text())
  }
  if err := scanner.Err(); err != nil {
    return err
  }
  return nil
}
```

#### Classifying Workflow

基于 Tensorflow 模型实现图像识别的主要流程如下：

- 图像转换 (Convert to tensor )
- 图像标准化( Normalize )
- 图像分类 （ Classifying )

```go
func recognizeHandler(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
  // Read image
  imageFile, header, err := r.FormFile("image")
  // Will contain filename and extension
  imageName := strings.Split(header.Filename, ".")
  if err != nil {
    responseError(w, "Could not read image", http.StatusBadRequest)
    return
  }
  defer imageFile.Close()
  var imageBuffer bytes.Buffer
  // Copy image data to a buffer
  io.Copy(&imageBuffer, imageFile)

  // ...

  tensor, err := makeTensorFromImage(&imageBuffer, imageName[:1][0])
  if err != nil {
    responseError(w, "Invalid image", http.StatusBadRequest)
    return
  }

  // ...
}
```

函数 makeTensorFromImage() which runs an image tensor through the normalization graph.
```go
func makeTensorFromImage(imageBuffer *bytes.Buffer, imageFormat string) (*tf.Tensor, error) {
  tensor, err := tf.NewTensor(imageBuffer.String())
  if err != nil {
    return nil, err
  }
  graph, input, output, err := makeTransformImageGraph(imageFormat)
  if err != nil {
    return nil, err
  }
  session, err := tf.NewSession(graph, nil)
  if err != nil {
    return nil, err
  }
  defer session.Close()
  normalized, err := session.Run(
    map[tf.Output]*tf.Tensor{input: tensor},
    []tf.Output{output},
    nil)
  if err != nil {
    return nil, err
  }
  return normalized[0], nil
}
```

函数 maketransformimagegraph() 将图形的像素值调整到 224x224，以符合模型输入参数要求。

```go
func makeTransformImageGraph(imageFormat string) (graph *tf.Graph, input, output tf.Output, err error) {
  const (
    H, W  = 224, 224
    Mean  = float32(117)
    Scale = float32(1)
  )
  s := op.NewScope()
  input = op.Placeholder(s, tf.String)
  // Decode PNG or JPEG
  var decode tf.Output
  if imageFormat == "png" {
    decode = op.DecodePng(s, input, op.DecodePngChannels(3))
  } else {
    decode = op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))
  }
  // Div and Sub perform (value-Mean)/Scale for each pixel
  output = op.Div(s,
    op.Sub(s,
      // Resize to 224x224 with bilinear interpolation
      op.ResizeBilinear(s,
        // Create a batch containing a single image
        op.ExpandDims(s,
          // Use decoded pixel values
          op.Cast(s, decode, tf.Float),
          op.Const(s.SubScope("make_batch"), int32(0))),
        op.Const(s.SubScope("size"), []int32{H, W})),
      op.Const(s.SubScope("mean"), Mean)),
    op.Const(s.SubScope("scale"), Scale))
  graph, err = s.Finalize()
  return graph, input, output, err
}
```

最后，将格式化的 image tensor 输入到 Inception model graph 中运算。

```go
session, err := tf.NewSession(graph, nil)
if err != nil {
  log.Fatal(err)
}
defer session.Close()
output, err := session.Run(
  map[tf.Output]*tf.Tensor{
    graph.Operation("input").Output(0): tensor,
  },
  []tf.Output{
    graph.Operation("output").Output(0),
  },
  nil)
if err != nil {
  responseError(w, "Could not run inference", http.StatusInternalServerError)
  return
}
```

#### Testing

```go
func main() {
  if err := loadModel(); err != nil {
    log.Fatal(err)
    return
  }
  r := httprouter.New()
  r.POST("/recognize", recognizeHandler)
  err := http.ListenAndServe(":8080", r)
  if err != nil {
    log.Println(err)
    return
  }
}
```

![识别案例：黑天鹅](http://riboseyim-qiniu.riboseyim.com/Tensorflow-demo-Black.png)

```json
$ curl localhost:8080/recognize -F 'image=@../data/IMG_3560.png'
{
  "filename":"IMG_3000.png",
  "labels":[
    {"label":"black swan","probability":0.98746836,"Percent":"98.75%"},
    {"label":"oystercatcher","probability":0.0040768473,"Percent":"0.41%"},
    {"label":"American coot","probability":0.002185003,"Percent":"0.22%"},
    {"label":"black stork","probability":0.0011524856,"Percent":"0.12%"},
    {"label":"redshank","probability":0.0010183558,"Percent":"0.10%"}]
}
```

![IMG_3560.png](http://riboseyim-qiniu.riboseyim.com/Tensorflow-Demo-Dog-1.png)

![IMG_3608.png](http://riboseyim-qiniu.riboseyim.com/Tensorflow-Demo-Dog-2.png)

通过上面的案例我们可以发现，这个服务目前可以对于黑天鹅图像的推算概率值为 98.75%，非常准确；但是对于另外两张宠物狗的图像，最高的推算概率值也仅有 30% 左右，虽然也没有被识别成猫咪或者狼，但是和理想效果要求可用性还有一段距离（此处暂时忽略物种本身的复杂性）。主要是因为现在我们使用的还只是一个非常“原始”的模型，如果需要为小众领域服务（宠物，也可以是其它事物），需要通过训练（Training Models）增强优化，或者引入更丰富的标签，更合适的模型。当然，训练过程中也会存在样本质量不佳的情况，错误样本和各种噪音也会影响准确度。

## 待续

#### **Ideas:勋章识别器**
#### **Ideas:美女打分器**

## 扩展阅读
- [How to Retrain Inception's Final Layer for New Categories](https://www.tensorflow.org/tutorials/image_retraining)

- [Getting Started with TensorFlow: A Machine Learning Tutorial](https://www.kdnuggets.com/2017/12/getting-started-tensorflow.html)
- **Youtube**[Introduction To TensorFlow](https://www.youtube.com/watch?v=FQ660T4uu7k)

- [We Need to Go Deeper: A Practical Guide to Tensorflow and Inception](https://medium.com/initialized-capital/we-need-to-go-deeper-a-practical-guide-to-tensorflow-and-inception-50e66281804f)

- [Tensorflow.org: Image Recognition](https://www.tensorflow.org/tutorials/image_recognition)
>We know that label 866 (military uniform) should be the top label for the Admiral Hopper image.

## 扩展阅读:[《The Machine Learning Master》](https://www.gitbook.com/book/riboseyim/machine-learning)
![](http://riboseyim-qiniu.riboseyim.com/banner-MLM-201803.png)

- [Machine Learning(一):基于 TensorFlow 实现宠物血统智能识别](https://riboseyim.github.io/2018/01/17/Machine-Learning-TensorFlow/)
- [Machine Learning (二):宠物智能识别之 Using OpenCV with Node.js](https://riboseyim.github.io/2018/01/15/Machine-Learning-OpenCV/)
- [Machine Learning:机器学习项目](https://riboseyim.github.io/2018/02/09/Machine-Learning-Projects/)
- [Machine Learning:机器学习算法](https://riboseyim.github.io/2018/02/10/Machine-Learning-Algorithms/)
- [Machine Learning:机器学习书单](https://riboseyim.github.io/2018/01/25/Machine-Learning-Books/)
- [Machine Learning:机器学习技术与知识产权法](https://riboseyim.github.io/2018/02/16/Machine-Learning-Law/)
- [Machine Learning:人工智能媒体报道集](https://riboseyim.github.io/2017/08/29/Machine-Learning-News)
- [数据可视化（三）基于 Graphviz 实现程序化绘图](https://riboseyim.github.io/2017/09/15/Visualization-Graphviz/)

## 参考文献
- [Understanding Tensorflow using Go](https://pgaleone.eu/tensorflow/go/2017/05/29/understanding-tensorflow-using-go/)
- [Using your tensorflow model with go](https://nilsmagnus.github.io/post/go-tensorflow/)
- [Build an Image Recognition API with Go and TensorFlow](https://outcrawl.com/image-recognition-api-go-tensorflow/)
- [TensorFlow系统架构及高性能程序设计](http://www.infoq.com/cn/articles/tensorflow-architecture-and-programming?utm_campaign=infoq_content&utm_source=infoq&utm_medium=feed&utm_term=global)
- [Yahoo开源TensorFlowOnSpark](http://www.infoq.com/cn/news/2017/02/Spark-Yahoo-TensorFlowOnSpark?utm_campaign=infoq_content&utm_source=infoq&utm_medium=feed&utm_term=global)
