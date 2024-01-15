# ADReSSo

This is the repo for the codes to fine-tuning the transformers with the [ADReSSo dataset](https://dementia.talkbank.org/ADReSSo-2021/). The dataset has been converted into [Hugging Face dataset](https://huggingface.co/datasets/nevikw39/ADReSSo).

Our goal is to resolve the first, classification task in the [2021 ADReSSo Challenge](https://luzs.gitlab.io/adresso-2021/) at [2021 INTERSPEECH](https://www.interspeech2021.org/special-sessions-challenges).

## Setup

### Franework Versions

- Transformers 4.35.1
- Pytorch 2.1.0+cu121
- Datasets 2.14.6
- Tokenizers 0.14.1

You can get these package with the following command:

    pip install transformers datasets evaluate accelerate

If you want to get more information, please refer to [requirements.txt](.\requirements.txt) and [environment.yaml](.\environment.yaml).

## Datasets

Our datasets are retrieved from DementiaBank. Due to the limit of the license, they must be kept in private.  
If you want to train our model in different datasets, here is the requirements.  
You need to use [hugging face dataset](https://huggingface.co/docs/datasets/index), and split them into train and test.  
Each dataset need to have the following features:

- audio: the audio transform by librosa
- label: label of data (1 means control group and 0 means Alzheimer’s Dementia)
- mmse(option): Mini–mental state examination

Here is an example.

    DatasetDict({
        train: Dataset({
            features: ['audio', 'label', 'mmse'],
            num_rows: 237
        })
        test: Dataset({
            features: ['audio', 'label', 'mmse'],
            num_rows: 46
        })
    })

## Model

### Execution Arguments

- -m: the model you want to train (must on huggingface)
- -d: sample duration (in seconds)
- -b: training batch size
- -g: Gradient Accumulation Steps
- -hp: enable half precision

|Argument|Type|Default value|
|-----|--------|----|
|-m|string|facebook/wav2vec2-base|
|-d|integer|30|
|-b|integer|8|
|-g|integer|4|
|-hp|boolean|False|

### Hyper-parameters Tuning Guide

$$\text{Equivalent Batch Size}=\\#\text{GPUs}\times\text{Batch Size Per GPU}\times\text{Gradient Accumulation Steps}$$

So if the model fails to fit in the GPU, i.e., CUDA out of memory, try to decrease $\text{Batch Size Per GPU}$ while $\text{Gradient Accumulation Steps}$ incresed simultaneously. In this manner, we trade off time for space.

### Prediction

You can refer to this [repository](https://github.com/NTHU-ML-2023-team19/ADReSSo-Notebooks)

### Notice

- You need to modify shebang of .py file.
- You need a hugging face dataset to support training.
- You need to create your hugging face repository to save your model.

## Results

### Classification

Acoustics
|Model Variant|Accuracy|F1|
|-------------|:------:|:-:|
|distil-whisper-large-v2| 0.8451| 0.8607|
|whisper-large-v3| 0.8451| 0.8406|
|distil-whisper-medium.en| 0.8169| 0.7936|
|whisper-medium| 0.7606| 0.7792|
|whisper-medium.en| 0.7324| 0.7324|

Linguistic
|Model Variant|Accuracy|F1|
|-------------|:------:|:-:|
|roberta-large| 0.8310| 0.8421|
|bart-large-mnli| 0.8028| 0.8000|
|bert-large| 0.7746| 0.7500|
|bart-large| 0.7465| 0.7500|
|flan-t5-large| 0.7465| 0.7273|

### Regression

|Model Variant|RMSE|
|-------------|:--:|
|whisper-medium.en| 4.5335|
|whisper-large-v3| 4.5682|
|distil-whisper-large-v2| 4.7742|
|whisper-medium| 4.8297|
|distil-whisper-medium.en| 4.9445|
