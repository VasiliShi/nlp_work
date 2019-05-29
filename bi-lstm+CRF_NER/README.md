### 中文词性标注bi-lstm的实现，

代码来自[这里](<https://github.com/Determined22/zh-NER-TF>),按照该repo又重新实现了一遍。该代码中实现的比较冗余不过很适合学习。新TF api的实现可以参考[这份代码](https://github.com/guillaumegenthial/tf_ner)，采用的`tf.estimator`和`tf.data`实现。

- 训练

  `python main.py  --mode=train`

- 演示

  `python main.py  --mode=demo --demo_model=1521112368`

