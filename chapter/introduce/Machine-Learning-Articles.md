## Machine Learning 资讯 | 机器学习应用案例

## 摘要
- 机器学习参考文集
- Machine Learning 资讯 | 基于深度学习识别姑息治疗患者
- Machine Learning Top Articles

## 基于深度学习识别姑息治疗患者
Stanford ML Group 建立了一个使用深度学习算法的程序，根据电子健康记录（Electronic Health Record ，EHR，包括病历、心电图、医疗影像等信息）数据确定在未来3-12个月高风险死亡的住院患者。这些病人的预警信息将发送给姑息治疗小组，这有助于姑息护理小组尽早介入、提供服务。

姑息治疗（Palliative Care ，在日本、中国台湾翻译为舒缓医学）起源于 hospice运动，最早起源于公元四世纪。根据世界卫生组织的定义，姑息治疗强调控制疼痛及患者有关症状，并对心理、社会和精神问题予以重视，目的是为病人和家属赢得最好的生活质量。

![](http://omb2onfvy.bkt.clouddn.com/ML-News-201802-3.png)

![](http://omb2onfvy.bkt.clouddn.com/ML-News-201802-2.png)

预测模型是一个 18 层的深度神经网络，输入参数为一个病人的 EHR 数据，输出为未来 3-12 个月死亡的概率。训练数据采用斯坦福医院 EHR 数据库中的历史数据，包含超过 200 万名患者的数据。EHR 数据包括患者过去 12 个月的诊断结论、治疗程序、处方和相关细节（经过脱敏和技术处理，以替代码的形式表示），所有数据被转换成 13654 维的特征向量。训练好的模型 AUROC 评分达到 0.93 ，交叉验证的平均精度为0.69 分。

对于机器学习系统来说，使用户可以根据预测结果采取行动，需要提供预测结果的详细解释，这点对于建立用户信心至关重要。Stanford 的程序可以自动生成一个报告，在病人的 EHR 数据中高亮突出对于预测结果具有重要影响因子的条目。

## 分类
- [图像处理 Image Manipulation](#)
- [风格转换 Style Transfer](#)
- [图像分类 Image Classification](#)
- [脸部识别 Face Recognition](#)
- [视频稳定化 Video Stabilization](#)
- [目标检测 Object Detection](#)
- [自动驾驶汽车 Self Driving Car](#)
- [智能推荐 Recommendation Al](#)
- [智能游戏 Gaming Al](#)
- [智能下棋 Chess Al](#)
- [智能医学 Medical Al](#)
- [智能演说 Speech Al](#)
- [智能音乐 Music Al](#)
- [自然语言处理 Natural Language Processing](#)
- [智能预测 Prediction](#)

<br>

Mybridge AI  在 20000 篇关于创建机器学习应用的文章中挑选了前 50 名。从有实践经验的数据科学家那里学习是一个好方法，我们可以的分享中获得构建、运营机器学习应用的经验教训。50 篇文章大致可以分为 15 个主题，如下所示：

### Recommended Learning

- [The Beginner’s Guide to Building an Artificial Intelligence in Unity.](http://bit.ly/2nbsc5n)
- [Deep Learning and Computer Vision A-Z™: Learn OpenCV, SSD & GANs and create image recognition apps.](http://bit.ly/2naZ4vg)

<br>

### 图像处理 Image Manipulation
* [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://tcwang0509.github.io/pix2pixHD?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Using Deep Learning to Create Professional-Level Photographs](https://research.googleblog.com/2017/07/using-deep-learning-to-create.html?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [High Dynamic Range (HDR) Imaging using OpenCV (Python)](https://www.learnopencv.com/high-dynamic-range-hdr-imaging-using-opencv-cpp-python?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)

<br>

### 风格转换 Style Transfer
* [Visual Attribute Transfer through Deep Image Analogy](https://arxiv.org/abs/1705.01088?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Deep Photo Style Transfer: A deep-learning approach to photographic style transfer that handles a large variety of image content while faithfully transferring the reference style](https://arxiv.org/abs/1703.07511?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)

<br>

### 图像分类 Image Classification
* [Feature Visualization: How neural networks build up their understanding of images](https://distill.pub/2017/feature-visualization?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [An absolute beginner's guide to Image Classification with Neural Networks](https://github.com/humphd/have-fun-with-machine-learning?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Background removal with deep learning](https://medium.com/towards-data-science/background-removal-with-deep-learning-c4f2104b3157?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)

<br>

### 人脸识别 Face Recognition
* [Large Pose 3D Face Reconstruction from a Single Image via Direct Volumetric CNN Regression](http://aaronsplace.co.uk/papers/jackson2017recon?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Eye blink detection with OpenCV, Python, and dlib](http://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [DEAL WITH IT in Python with Face Detection](https://www.makeartwithpython.com/blog/deal-with-it-generator-face-recognition?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)

<br>

### 视频稳定化 Video Stabilization
* [Fused Video Stabilization on the Pixel 2 and Pixel 2 XL](https://research.googleblog.com/2017/11/fused-video-stabilization-on-pixel-2.html?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)

<br>

### 目标检测 Object Detection
* [How HBO’s Silicon Valley built “Not Hotdog” with mobile TensorFlow, Keras & React Native](https://hackernoon.com/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Object detection: an overview in the age of Deep Learning](https://tryolabs.com/blog/2017/08/30/object-detection-an-overview-in-the-age-of-deep-learning?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [How to train your own Object Detector with TensorFlow’s Object Detector API](https://medium.com/towards-data-science/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Real-time object detection with deep learning and OpenCV](http://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)

<br>

### 自动驾驶汽车 Self Driving Car
* [Self-driving Grand Theft Auto V with Python : Intro [Part I] - Sentdex](https://www.youtube.com/watch?v=ks4MPfMq8aQ?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Recognizing Traffic Lights With Deep Learning: How I learned deep learning in 10 weeks and won $5,000](https://medium.freecodecamp.com/recognizing-traffic-lights-with-deep-learning-23dae23287cc?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)

<br>

### 智能推荐 Recommendation AI
* [Spotify’s Discover Weekly: How machine learning finds your new music](https://hackernoon.com/spotifys-discover-weekly-how-machine-learning-finds-your-new-music-19a41ab76efe?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Artwork Personalization at Netflix](https://medium.com/netflix-techblog/artwork-personalization-c589f074ad76?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)

<br>

### 智能游戏 Gaming AI
* [MariFlow - Self-Driving Mario Kart w/Recurrent Neural Network](https://www.youtube.com/watch?v=Ipi40cb_RsI?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [OpenAI Baselines: DQN. Reproduce reinforcement learning algorithms with performance on par with published results.](https://blog.openai.com/openai-baselines-dqn?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Reinforcement Learning on Dota 2 [Part II]](https://blog.openai.com/more-on-dota-2?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Creating an AI DOOM bot](https://www.codelitt.com/blog/doom-ai?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Phase-Functioned Neural Networks for Character Control](http://theorangeduck.com/page/phase-functioned-neural-networks-character-control?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [The Game Imitation: Deep Supervised Convolutional Networks for Quick Video Game AI - Stanford University](https://arxiv.org/abs/1702.05663?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Introducing: Unity Machine Learning Agents – Unity Blog](https://blogs.unity3d.com/2017/09/19/introducing-unity-machine-learning-agents?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)

<br>

### 智能下棋 Chess AI
* [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [AlphaGo Zero: Learning from scratch | DeepMind](https://deepmind.com/blog/alphago-zero-learning-scratch?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [How Does DeepMind's AlphaGo Zero Work?](https://www.youtube.com/watch?v=vC66XFoN4DE?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [A step-by-step guide to building a simple chess AI](https://medium.freecodecamp.com/simple-chess-ai-step-by-step-1d55a9266977?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)

<br>

### 智能医学 Medical AI
* [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning](https://stanfordmlgroup.github.io/projects/chexnet?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Can you improve lung cancer detection? 2nd place solution for the Data Science Bowl 2017.](http://juliandewit.github.io/kaggle-ndsb2017?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Improving Palliative Care with Deep Learning - Andrew Ng](https://stanfordmlgroup.github.io/projects/improving-palliative-care?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Heart Disease Diagnosis with Deep Learning](https://blog.insightdatascience.com/heart-disease-diagnosis-with-deep-learning-c2d92c27e730?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)

<br>

### 智能演说 Speech AI
* [Tacotron: A Fully End-to-End Text-To-Speech Synthesis Model - Data Scientists at Google](https://arxiv.org/abs/1703.10135?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Sequence Modeling with CTC](https://distill.pub/2017/ctc/?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Deep Voice: Real-time Neural Text-to-Speech - Baidu Silicon Valley AI Lab](https://arxiv.org/abs/1702.07825?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Deep Learning for Siri’s Voice: On-device Deep Mixture Density Networks for Hybrid Unit Selection Synthesis - Apple](https://machinelearning.apple.com/2017/08/06/siri-voices.html?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)

<br>

### 智能音乐 Music AI
* [Computer evolves to generate baroque music!](https://www.youtube.com/watch?v=SacogDL_4JU?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Make your own music with WaveNets: Making a Neural Synthesizer Instrument](https://magenta.tensorflow.org/nsynth-instrument?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)

<br>

### 自然语言处理 Natural Language Processing
* [Learning to communicate: Agents developing their own language - OpenAI Research](https://openai.com/blog/learning-to-communicate?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Big Picture Machine Learning: Classifying Text with Neural Networks and TensorFlow](https://medium.freecodecamp.com/big-picture-machine-learning-classifying-text-with-neural-networks-and-tensorflow-d94036ac2274?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [A novel approach to neural machine translation - Facebook AI Research](https://code.facebook.com/posts/1978007565818999/a-novel-approach-to-neural-machine-translation?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [How to make a racist AI without really trying](https://blog.conceptnet.io/2017/07/13/how-to-make-a-racist-ai-without-really-trying?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)

<br>

### 预测 Prediction
* [Using Machine Learning to Predict Value of Homes On Airbnb](https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Engineering Uncertainty Estimation in Neural Networks for Time Series Prediction at Uber](https://eng.uber.com/neural-networks-uncertainty-estimation?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [Using Machine Learning to make parking easier](https://research.googleblog.com/2017/02/using-machine-learning-to-predict.html?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
* [How to Predict Stock Prices Easily - Intro to Deep Learning #7](https://www.youtube.com/watch?v=ftMq5ps503w?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)

## 扩展阅读:[《The Machine Learning Master》](https://www.gitbook.com/book/riboseyim/machine-learning)
![](http://p11slcnom.bkt.clouddn.com/banner-MLM-201803.png)
- [Machine Learning(一):基于 TensorFlow 实现宠物血统智能识别](https://riboseyim.github.io/2018/01/17/Machine-Learning-TensorFlow/)
- [Machine Learning(二):宠物智能识别之 Using OpenCV with Node.js](https://riboseyim.github.io/2018/01/15/Machine-Learning-OpenCV/)
- [Machine Learning:机器学习项目](https://riboseyim.github.io/2018/02/09/Machine-Learning-Projects/)
- [Machine Learning:机器学习算法](https://riboseyim.github.io/2018/02/10/Machine-Learning-Algorithms/)
- [Machine Learning:机器学习书单](https://riboseyim.github.io/2018/01/25/Machine-Learning-Books/)
- [Machine Learning:机器学习技术与知识产权法](https://riboseyim.github.io/2018/02/16/Machine-Learning-Law/)
- [Machine Learning:人工智能媒体报道集](https://riboseyim.github.io/2017/08/29/Machine-Learning-News)
- [数据可视化（三）基于 Graphviz 实现程序化绘图](https://riboseyim.github.io/2017/09/15/Visualization-Graphviz/)
