# CuteChineseTTS

 - 摘要
     - 这是一个极其精简的中文语音合成系统，借鉴自 [Merlin](https://github.com/CSTR-Edinburgh/merlin)，但删除了庞杂的 lab 处理系统，可供教学、科研和商业使用
     - 提供了预训练的声学模型，可直接从拼音合成音频
     - 提供了一整套特征提取和模型训练方法
 
 - 特别感谢标贝科技开源的 10000 句中文 TTS [标注数据](http://www.data-baker.com/open_source.html)
 
 - 依赖环境
     - Ubuntu 16.04
     - Python3
     - CNTK2.6
 
 - 使用预训练的模型
     - 本项目使用前 3200 句话预训练了一个声学模型，可直接使用
     - pipeline.py 给出了从拼音到音频的主要流程。如果环境设置正确，那么运行 pipeline.py 就会合成一句中文
     - 为了简化，本项目未通过模型推断音素的时长，而是使用了从样本中统计的结果
     
 - 从头开始训练模型
     - 1、下载并解压数据，本项目推荐的 [数据] 目录结构见附件

     - 2、提取参数
         - 修改 cute/common/parameter.py 中的参数
         - 运行 cute/preprocess/parse_interval.py 处理标注的音素和时长
         - 运行 cute/preprocess/generate_label.py 生成 lab 文件
         - 运行 cute/preprocess/feat_extractor.py 提取声学参数
         - 运行 cute/preprocess/normalize_feat.py 归一化声学参数
         
     - 3、训练声学模型
         - 运行 cute/train/train.py 训练声学模型
         
     - 4、测试声学模型
         - 运行 cute/pipeline.py 合成声音

 - 作者声明
     - 本项目提供了一个可供教学、科研或商用的极简中文语音合成系统
     - 无论是在效果还是在速度上，本项目的很多模块明显可以优化，如时长模型、声学模型、合成器等
     - 欢迎大家共享文本规整化（Text Normalization）或者注音模块到本项目，共同推动行业发展
     - 如有大规模商业使用计划，欢迎同作者交流解决方案

 - 目录结构
 
        baker
        ├── ProsodyLabeling
        │   │ 	└── 000001-010000.txt
        ├── PhoneLabeling
        │   │ 	├── 000001.interval
        │   │ 	├── 000002.interval
        │   │ 	├── ...
        │   │ 	└── 010000.interval
        ├── Wave
        │   │ 	├── 000001.wav
        │   │ 	├── 000002.wav
        │   │ 	├── ...
        │   │ 	└── 010000.wav
        ├── Lab
        ├── 48k
        │   │ 	├── features
        │   │ 	│   │ 	├── bap
        │   │ 	│   │ 	├── lf0
        │   │ 	│   │ 	├── log
        │   │ 	│   │ 	├── mgc
        │   │ 	│   │ 	├── norm
        │   │ 	│   │ 	├── tmp
        │   │ 	│   │ 	├── features.norm
        │   │ 	├── wav
        │   │ 	│   │ 	├── *.wav
        ├── ph_dur.pkl
        └── ph_set.json

