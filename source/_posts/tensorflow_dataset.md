---
title: Dataset 
date: 2018-10-10
author : ZJun
---



# Dataset简介
Dataset 简而言之就是做数据准备工作，将数据读取之后做规范化的整合以供模型输入。

这个环节处理的优劣直接决定了后续特征处理的效率和模型训练的速度。

我们看下官方对于Dataset的描述

> A `Dataset` can be used to represent an input pipeline as a collection of elements (nested structures of tensors) and a "logical plan" of transformations that act on those elements.

# Dataset 数据读取
Dataset支持各种数据的读取，总的可以分为两类，内存读取和文件读取。内存读取一般就是读取 NumPy arrays，文件读取可以分为TFRecord和Text。

## 内存读取
### tf.data.Dataset.from_tensor_slices

如果你的数据能够全部读进内存，那么创建Dataset最简单的方法就是使用`tf.data.Dataset.from_tensor_slices`，

```python
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

dataset = tf.data.Dataset.from_tensor_slices((features, labels))
```


## 文件读取
### tf.data.TextLineDataset

当数据量比较多的时候，读入内存是不现实的，这个时候我们通过`tf.data.TextLineDataset`直接读取文件来构建Dataset

```python
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]
dataset = tf.data.TextLineDataset(filenames)
```

此处`filenames` 是一个list，也就是说`tf.data.TextLineDataset`可以接受任意数量的文件作为输入，为此我们需要一个获取数据目录下所有文件名的操作，在此推荐使用`tf.gfile`

```python
if tf.gfile.IsDirectory(data_file):
    file_name = [f for f in tf.gfile.ListDirectory(data_file) if not f.startswith('.')]
    data_file_list = [data_file + '/' + f for f in file_name]
else:
    data_file_list = [data_file]
```

### tf.data.TFRecordDataset

TFRecord 文件格式是一种面向记录的简单二进制格式，很多 TensorFlow 应用采用此格式来训练数据。对于这种数据我们使用`tf.data.TFRecordDataset`来处理

```python
# Creates a dataset that reads all of the examples from two files.
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
```



# Dataset 数据操作

通过上一节介绍的API读入数据后通常还需要进一步处理以方便后续使用，我们从数据处理和数据准备两个角度来介绍

## 数据处理

### 解析CSV文件

### 解析tfrecord文件


## 数据准备

- shuffle
- repeat
- batch

# Dataset 数据输出
- 输出结构和后期运用介绍

# 进阶补充

## 构建TFRecord文件

此处需要另外说明的就是如何构建TFRecord文件

```python
# config file
config.FIELD_DELIM = ','
config.HEADER = ['user_id','label']
config.RECORD_DEFAULTS = ['',0]
```

```python
def _int64_feature(value):
    value = int(value)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    value = float(value)
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    value = value.encode('utf8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _add_feature(value, record_default):
    if isinstance(record_default, float):
        return _float_feature(value)
    elif isinstance(record_default, int):
        return _int64_feature(value)
    elif isinstance(record_default, str):
        return _bytes_feature(value)
    
def writeTFRecord(raw_data_path_list,tfrecord_file_name):
 
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(tfrecord_file_name)

    for data_file_path in raw_data_path_list:
        print('precess {}'.format(data_file_path))
        with open(data_file_path, 'r') as f:
            for line in f.readlines():
                split_line = line.strip().split(config.FIELD_DELIM)
                # Create a feature
                feature = {}
                index = 0
                for col_name, record_default in zip(config.HEADER, config.RECORD_DEFAULTS):
                    feature.update({col_name: _add_feature(split_line[index],
                                                           record_default)})
                    index += 1
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

    writer.close()
```

写出TFRecord的过程中，我们会生成protocol buffer，数据样例如下

```protobuf
features {
  feature {
    key: "label"
    value {
      int64_list {
        value: 0
      }
    }
  }
  feature {
    key: "user_id"
    value {
      bytes_list {
        value: "60821417"
      }
    }
  }
}
```



## 自定义map处理函数

## 性能调优：prefetch



