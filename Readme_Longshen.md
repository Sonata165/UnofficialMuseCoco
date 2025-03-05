
## Intro

- 目前的代码用来对musecoco做sequence to sequence finetununing。稍微比MuseCoco原本的代码整齐一点点，方便换成自己的数据做finetune。
- 如何用自己的数据finetune呢？(fairseq)
    - 把数据集整理成这个样子：![Alt text](examples/image.png)。[split]_input.txt是source side sequences, [split].txt是target side sequence. 每个文件都有num_sample行。相同行号的数据是对应的input和output。如果要从midi files准备训练数据，可以参考我的流程，在dataset_preparation/create_midi_only_dataset.py的MidiFeatureExtraction.process()和TwoBarDatasetPreparation.process()
    - 准备好dict.txt  这个文件是原模型的dictionary，表示怎么把remi token变成id。如果需要新增一些token，可以选一些目前模型用不上的token，换成你新增的token。对这个文件的更改不能改变行数。
    - 运行dataset_preparation/preprocess.sh. 运行前需要指定dict.txt的位置和数据位置。
    - 可以训练了。src/train_large.py是一个python的训练入口，方便debug。训练时只会用到training和validation set。
    - 训练结束后，运行src/infer_large.py，会使用test set的condition进行generation。
    - 可以使用src/eval_lm.py来测试训好的模型的perplexity（不过可能没啥用）
    - src/evaluate.py是我的task使用的一些客观指标。仅供参考。
- 如何finetune?(hugging face)
    - 运行src_hf/lightning_train.py 比如
        CUDA_VISIBLE_DEVICES=0 python lightning_train.py hparams/[path_to_your_hparam]
    - 需要根据你的需求修改三个
- 这个版本和musecoco的模型主要区别
    - 这个版本的Tokenizer的方式不含velocity, speed和time signature。
    - original ver不对condition计算positional encoding. 这个版本会，会导致直接inference结果稍有区别。
    - original ver计算loss时没考虑condition那部分。这个版本的loss是整个sequence (condition <sep> target) 的CE loss。
    - 在src/linear_mask/A2M_task_new.py里增加了ConditionDataset class, 是修改了原本的CommandDataset来添加data augmentation
    - 很难看懂原版的fairseq模型是怎么组织input的（for conditional generation）。在src_hf/inference_demo.py里的实现有些是我的猜测，效果不一定好。
- 其他事项
    - fp16不work，有nan。原版Attention的实现没考虑数值范围问题。
    - Note: hugging face version only support generating for one sample at a time (no batch generation)
    - Note2: 实现的时候为了方便，positional encoding的计算方式和fairseq有所不同。 
    - huggingface version没有给prefix计算positional encoding，但是原本的fairseq版本是给prefix计算positional encoding的。
    - 但可以保证除此之外，huggingface version和fairseq version的forward pass完全一致



## Checkpoint

```bash
gdown https://drive.google.com/uc?id=1HJvrOi_cli48RDm7ni5VAafVEi8qcTGN  # Download the checkpoint (1B model)
200M model在微信群里

# Environment
conda create -n musecoco python=3.8

# PyTorch (CUDA 12.1 support, GPU6 server, installed nvcc 11.7)
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# Dependencies
pip install -r requirements_feat_to_music.txt   

# Fix dependencies
pip install tensorboardx==2.1 protobuf==3.20 # EOFError still exist at end of training
pip install tensorboardx --upgrade

# Do the inference with the first 50 samples in the test set
bash interactive_1billion.sh 0 50    

bash interactive_1billion.sh 0 1  

# Archived (command does not work)
(in python 3.9)
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia # cuda too new
pip install --upgrade pip setuptools wheel  
pip install -v --no-cache-dir fairseq==0.10.2 # not working
pip install fairseq==0.10.0 # cannot install
pip install fairseq==0.12.0 # cannot install
pip install  -v --no-cache-dir fairseq==0.12.1
pip install fairseq==0.12.2 # can install, but cannot run
```

## MuseCoco Input Details
Here are possible input tokens to MuseCoco
    "I1s2": "Instrument", a 28-dim multi-hot vector. ("乐器个列表，每个列表长度为3，依次为是、否、NA")
        Format: I1s2_[inst_id]_0  
        inst_id: [0-27]
        There are condition tokens like I1s2_[inst_id]_1 and ..._2. Deprecated for simplicity.
    "R1": "Rhythm Danceability", 
        R1_0: dancable
        R1_1: not dancable
        R1_2: NA
    "R3": "Rhythm Intensity",  
        R3_0: not intense
        R3_1: med
        R3_2: intense
    "S2s1": "Artist", 
        Format: S2s1_[id]
        id's value:
            'beethoven': 0,
            'mozart': 1,
            'chopin': 2,
            'schubert': 3,
            'schumann': 4,
            'bach-js': 5,
            'haydn': 6,
            'brahms': 7,
            'Handel': 8,
            'tchaikovsky': 9,
            'mendelssohn': 10,
            'dvorak': 11,
            'liszt': 12,
            'stravinsky': 13,
            'mahler': 14,
            'prokofiev': 15,
            'shostakovich': 16,
    "S4": "Genre",
        S4_[gid]_0
        gid's value:
            'New Age': 0,
            'Electronic': 1,
            'Rap': 2,
            'Religious': 3,
            'International': 4,
            'Easy_Listening': 5,
            'Avant_Garde': 6,
            'RnB': 7,
            'Latin': 8,
            'Children': 9,
            'Jazz': 10,
            'Classical': 11,
            'Comedy_Spoken': 12,
            'Pop_Rock': 13,
            'Reggae': 14,
            'Stage': 15,
            'Folk': 16,
            'Blues': 17,
            'Vocal': 18,
            'Holiday': 19,
            'Country': 20,
            'Symphony': 21,
        Similarly, S4_[gid]_1 and S4_[gid]_2 are deprecated.
    "B1s1": "Bar", represent bar个数区间的id
        B1s1_[bid]
        bid's value:
            0：1-4，
            1：5-8，
            2：9-12，
            3：13-16
    "TS1s1": "Time Signature",
        TS1s1_[tsid]
        tsid's value:
            0: (4, 4), 
            1: (2, 4), 
            2: (3, 4), 
            3: (1, 4), 
            4: (6, 8), 
            5: (3, 8)
    "K1": "Key",
        K1_0: major
        K1_1: minor
        K1_2: unknown
    "T1s1": "Tempo",
        T1s1_[tid]
        tid's value:
            0表示慢，
            1表示适中
            2表示快。
    "P4": "Pitch Range", n_octaves
        P4_[0-12]
        0个8度，1个8度，...，11个8度, NA
    "EM1": "Emotion", but don't know the mapping. Detail not specified. Deprecated.
        EM1_[0-4] 
    "TM1": “Time", output duration in seconds (Deprecated)
        TM1_[0-5]
            0表示(0-15]秒，
            1表示(15-30]秒，
            2表示30-45秒，
            3表示45-60秒，
            4表示60秒以上

## Env configure

    conda create -n musecoco_hf python=3.8
    conda activate musecoco_hf
    
    # Install pytorch (GPU6) torch 2.0.1 and cuda 11.7
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 

    # Install pytorch (GPU2)    GPU3: 2.0.0+cu117
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

    # Prepare for install of pytorch-fast-transformers (if system cuda is not compatible with torch cuda)
    # Problem when install pytorch-fast-transformers:
    # This package will build something use system condatoolkit
    # To fix it by using conda in my env instead, do
    conda install cudatoolkit-dev -c conda-forge

    # Install fast transformers
    pip install git+https://github.com/idiap/fast-transformers.git

    # Install dependencies
    cd transformers # the transformer with MuseCoco integration
    pip install -e .

    pip install lightning miditoolkit pretty_midi pydub librosa absl-py music21 protobuf==4.25 tensorboard


## Input specification:
Note sequence: (o i p d). If position is same as previous note, (i p d). i cannot omit.

## Intro

- 目前的代码用来对musecoco做sequence to sequence finetununing。稍微比MuseCoco原本的代码整齐一点点，方便换成自己的数据做finetune。
- 如何用自己的数据finetune呢？
    - 把数据集整理成这个样子：![Alt text](examples/image.png)。[split]_input.txt是source side sequences, [split].txt是target side sequence. 每个文件都有num_sample行。相同行号的数据是对应的input和output。如果要从midi files准备训练数据，可以参考我的流程，在dataset_preparation/create_midi_only_dataset.py的MidiFeatureExtraction.process()和TwoBarDatasetPreparation.process()
    - 准备好dict.txt  这个文件是原模型的dictionary，表示怎么把remi token变成id。如果需要新增一些token，可以选一些目前模型用不上的token，换成你新增的token。对这个文件的更改不能改变行数。
    - 运行dataset_preparation/preprocess.sh. 运行前需要指定dict.txt的位置和数据位置。
    - 可以训练了。src/train_large.py是一个python的训练入口，方便debug。训练时只会用到training和validation set。
    - 训练结束后，运行src/infer_large.py，会使用test set的condition进行generation。
    - 可以使用src/eval_lm.py来测试训好的模型的perplexity（不过可能没啥用）
    - src/evaluate.py是我的task使用的一些客观指标。仅供参考。
- 这个版本和musecoco的模型主要区别
    - 这个版本的Tokenizer的方式不含velocity, speed和time signature。
    - original ver不对condition计算positional encoding. 这个版本会。
    - original ver计算loss时没考虑condition那部分。这个版本的loss是整个sequence (condition <sep> target) 的CE loss。
    - 在src/linear_mask/A2M_task_new.py里增加了ConditionDataset class, 是修改了原本的CommandDataset来添加data augmentation
- 其他事项
    - fp16不work，有nan。Attention的实现没考虑数值范围问题。



## Checkpoint

```bash
gdown https://drive.google.com/uc?id=1HJvrOi_cli48RDm7ni5VAafVEi8qcTGN  # Download the checkpoint (1B model)
200M model在微信群里

# Environment
conda create -n musecoco python=3.8

# PyTorch (CUDA 12.1 support, GPU6 server, installed nvcc 11.7)
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# Dependencies
pip install -r requirements_feat_to_music.txt   

# Fix dependencies
pip install tensorboardx==2.1 protobuf==3.20 # EOFError still exist at end of training
pip install tensorboardx --upgrade

# Do the inference with the first 50 samples in the test set
bash interactive_1billion.sh 0 50    

bash interactive_1billion.sh 0 1  

# Archived (command does not work)
(in python 3.9)
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia # cuda too new
pip install --upgrade pip setuptools wheel  
pip install -v --no-cache-dir fairseq==0.10.2 # not working
pip install fairseq==0.10.0 # cannot install
pip install fairseq==0.12.0 # cannot install
pip install  -v --no-cache-dir fairseq==0.12.1
pip install fairseq==0.12.2 # can install, but cannot run
```

## MuseCoco Input Details
Here are possible input tokens to MuseCoco
    "I1s2": "Instrument", a 28-dim multi-hot vector. ("乐器个列表，每个列表长度为3，依次为是、否、NA")
        Format: I1s2_[inst_id]_0  
        inst_id: [0-27]
        There are condition tokens like I1s2_[inst_id]_1 and ..._2. Deprecated for simplicity.
    "R1": "Rhythm Danceability", 
        R1_0: dancable
        R1_1: not dancable
        R1_2: NA
    "R3": "Rhythm Intensity",  
        R3_0: not intense
        R3_1: med
        R3_2: intense
    "S2s1": "Artist", 
        Format: S2s1_[id]
        id's value:
            'beethoven': 0,
            'mozart': 1,
            'chopin': 2,
            'schubert': 3,
            'schumann': 4,
            'bach-js': 5,
            'haydn': 6,
            'brahms': 7,
            'Handel': 8,
            'tchaikovsky': 9,
            'mendelssohn': 10,
            'dvorak': 11,
            'liszt': 12,
            'stravinsky': 13,
            'mahler': 14,
            'prokofiev': 15,
            'shostakovich': 16,
    "S4": "Genre",
        S4_[gid]_0
        gid's value:
            'New Age': 0,
            'Electronic': 1,
            'Rap': 2,
            'Religious': 3,
            'International': 4,
            'Easy_Listening': 5,
            'Avant_Garde': 6,
            'RnB': 7,
            'Latin': 8,
            'Children': 9,
            'Jazz': 10,
            'Classical': 11,
            'Comedy_Spoken': 12,
            'Pop_Rock': 13,
            'Reggae': 14,
            'Stage': 15,
            'Folk': 16,
            'Blues': 17,
            'Vocal': 18,
            'Holiday': 19,
            'Country': 20,
            'Symphony': 21,
        Similarly, S4_[gid]_1 and S4_[gid]_2 are deprecated.
    "B1s1": "Bar", represent bar个数区间的id
        B1s1_[bid]
        bid's value:
            0：1-4，
            1：5-8，
            2：9-12，
            3：13-16
    "TS1s1": "Time Signature",
        TS1s1_[tsid]
        tsid's value:
            0: (4, 4), 
            1: (2, 4), 
            2: (3, 4), 
            3: (1, 4), 
            4: (6, 8), 
            5: (3, 8)
    "K1": "Key",
        K1_0: major
        K1_1: minor
        K1_2: unknown
    "T1s1": "Tempo",
        T1s1_[tid]
        tid's value:
            0表示慢，
            1表示适中
            2表示快。
    "P4": "Pitch Range", n_octaves
        P4_[0-12]
        0个8度，1个8度，...，11个8度, NA
    "EM1": "Emotion", but don't know the mapping. Detail not specified. Deprecated.
        EM1_[0-4] 
    "TM1": “Time", output duration in seconds (Deprecated)
        TM1_[0-5]
            0表示(0-15]秒，
            1表示(15-30]秒，
            2表示30-45秒，
            3表示45-60秒，
            4表示60秒以上

## Env configure

    conda create -n musecoco_hf python=3.8
    
    # Install pytorch (GPU6) torch 2.0.1 and cuda 11.7
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 

    # Install pytorch (GPU3)
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

    # Prepare for install of pytorch-fast-transformers
    # Problem when install pytorch-fast-transformers:
    # This package will build something use system condatoolkit
    # To fix it by using conda in my env instead, do
    conda install cudatoolkit-dev -c conda-forge

    # Install fast transformers
    pip install git+https://github.com/idiap/fast-transformers.git

    # Install dependencies
    cd transformers # the transformer with MuseCoco integration
    pip install -e .

    pip install lightning miditoolkit pretty_midi pydub librosa absl-py music21  protobuf==4.25 
