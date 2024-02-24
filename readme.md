## Get Started

#### 1. Packages required in the virtual environment:

- Python == 3.6.9
- altair == 4.1.0
- streamlit == 1.6.0
- torch == 1.7.0+cpu
- transformers == 3.1.0

#### 2. Files in the folder:

- pretrained_model: folder that stores the pre-trained GPT-2 models
- vocab: folder that stores the vocabulary files of corresponding models
- save: folder that stores the generate results
- ui.py: the main program 

#### 3. How to run the program (suppose your virtual environment is called ”NLP”):

```
>> activate NLP
>> streamlit run ui_new.py
```

## Pre-trained Models

 gpt2-chinese-poem https://huggingface.co/uer/gpt2-chinese-poem

gpt2-chinese-lyric https://huggingface.co/uer/gpt2-chinese-lyric

gpt2-chinese-couplet https://huggingface.co/uer/gpt2-chinese-couplet

gpt2-chinese-ancient https://huggingface.co/uer/gpt2-chinese-ancient

gpt2-distil-chinese-cluecorpussmall https://huggingface.co/uer/gpt2-distil-chinese-cluecorpussmall

gpt2-prose-model (key: fpyu) https://pan.baidu.com/share/init?surl=nbrW5iw34GRhoTin8uU2tQ