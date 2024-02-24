# -*- coding:utf-8 -*-
# UI module, implemented by Streamlit
# CHEN SHUAI, Macau University of Science and Technology

# This Python file is used for text generation

# 0. Packages & Modules
# 包和模块
import os
import re 
import time
import random
import torch
import argparse
import altair as alt
import pandas as pd
import streamlit as st
import torch.nn.functional as F

from datetime import datetime
from torch import Tensor
from transformers import BertTokenizer
from transformers import GPT2LMHeadModel
from transformers import TextGenerationPipeline
from transformers import set_seed


# 1. 预备工作

# 1.1 Setting WebPage Title
# 1.1 设定网页标题
st.set_page_config(page_title="Chinese NLP ToolBox", layout="wide", initial_sidebar_state="expanded")


# 1.2 Configuring GPU or CPU device
# 1.2 配置 GPU或CPU
device_ids = 0

# arrange GPU devices according to "PCI_BUS_ID" order
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# set the GPU device currently used as "device_ids"
# note: should convert the digit 0 to string "0"
os.environ["CUDA_VISIBLE_DEVICE"] = str(device_ids)

# check whether the CUDA is available (if "YES", which means there is a GPU device)
# YES: device = torch.device("cuda")
# NO: device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() and int(device_ids) >= 0 else "cpu")


# 2. Other Functions

# 2.1 Model Settings Function
# 2.1 模型设置函数

# Setting Model, Tokenizer and Relative Parameters
# 设定模型、分词器以及相关参数

# device : select GPU or CPU
# model_path : the location of pre-trained model folder
# vocab_path : the location of vocabulary
# model.train() : set model in training mode (Switch ON Batch Normalization & Dropout) 
# model.train()用于模型训练之前
# model.eval() : set model in evaluation mode (Switch OFF Batch Normalization & Dropout)
# model.eval()用于模型验证之前

@st.cache()
def setting_model(device, model_path, vocab_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    model.to(device)
    model.eval()

    return model, tokenizer



# 2.2 result save function
# 2.2 生成结果保存函数
def result_save(specific_task, process_result, save):
    # get current time
    dt = datetime.now()
    dirs = './save'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    time_now = str(dt.year) + '-' + str(dt.month) + '-' + str(dt.day) + '-' + str(dt.hour) + '-' + str(dt.minute) + '-' + str(dt.second)
    file_name = "save/" + str(specific_task) + '-' + time_now + ".txt"

    if (save==True):
        with open(file_name, 'w') as file_object:
            file_object.write(process_result)

    return file_name


# 3. Decoding Module
# based on "GPT-2 Chinese" in Github & Transformers Official Documents

# 3.1 sampling sequence
# 3.1 精简解码模块

def decoding(model, content_ids, length, tokenizer, temperature, top_k, top_p, repetition_penalty, device, no_symbol):
    content_id_tensor = torch.tensor(content_ids, dtype=torch.long, device=device)
    content_id_tensor = content_id_tensor.unsqueeze(0)
    generated_id_tensor = content_id_tensor
    with torch.no_grad():
        for _ in range(length):
            # make the input tensor become a dictionary
            # before: generated_id_tensor = tensor([6821, 1126, 1921, 1921, 3698, 2523,  679, 7231])
            # after: input = {'input_ids': tensor([[6821, 1126, 1921, 1921, 3698, 2523,  679, 7231]])}
            # if raw text processed by tokenizer() directly, it would generate a dictionary with: (e.g.)
            # 1. 'input_ids': tensor([[ 101, 6821, 1126, 1921, 1921, 3698, 2523,  679, 7231,  102]]), 
            # 2. 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
            # 3. 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
            inputs = {'input_ids': generated_id_tensor[0][:].unsqueeze(0)}

            # pass the input to the model

            # CH：如果loss不是None的话，outputs是含有3个元素的元组，第一个元素是loss
            # len(outputs) == 3:
            # return values of outputs (if loss is not None):
            # 1. type:tensor; value: "loss"
            # 2. type:tensor; value: "lm_logits"
            # 3. type:tuple; value: transformer_output[1:]
            
            # CH：如果loss是None的话，outputs是含有2个元素的元组，而且第一个元素是logits
            # len(outputs) == 2:
            # return values of outputs (if loss is None):
            # 1. type:tensor; value: "lm_logits"
            # 2. type:tuple; value: transformer_output[1:]

            # CH：可以看到上面没有取transformer_output[0]，而这个是last_hidden_state
            # Notice 1:
            # transformer_output[0] is "last_hidden_state"
            # last_hidden_state is the "sequence of hidden-states at the output of the last layer of the model"
            # 
            # Notice 2.
            # "transformer_outputs" is the return value of function "self.transformer()"
            # while "self.transformer" is the return value of "GPT2Model(config)"
            outputs = model(**inputs)

            # CH：
            # outputs[0]是一个tensor，形状是[1,输入序列的长度,词汇表大小]，
            # 因此next_token_logits是输入序列最后一个字的logits
            # Here, the size of outputs[0], or the lm_logits is [1,seq_len,vocab_size] 
            # so "next_token_logits" is the logits of the last word in input sequence
            next_token_logits = outputs[0][0, -1, :]

            # CH：建立一个集合，generated_id_tensor是一个tensor（torch.Size([1, 输入序列长度]）
            # for 循环里面得id用于取出一个tensor（torch.Size([输入序列长度])）
            # generated_id_tensor is a tensor, whose size is torch.Size([1, seq_len])
            for id in set(generated_id_tensor):
                next_token_logits[id] /= repetition_penalty
            
            # CH：logits除以温度值
            next_token_logits = next_token_logits / temperature
            # CH：将UNK对应id的logits设置为负无穷
            # set the logits of token "UNK" as minus infinity
            # next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = float(0)
            next_token_logits[tokenizer.convert_tokens_to_ids('[SEP]')] = float(0)
            next_token_logits[tokenizer.convert_tokens_to_ids('[CLS]')] = float(0)
            next_token_logits[tokenizer.convert_tokens_to_ids('[PAD]')] = float(0)
            next_token_logits[tokenizer.convert_tokens_to_ids('[MASK]')] = float(0)

            # 禁止输出标点符号
            if (no_symbol == True):
                next_token_logits[tokenizer.convert_tokens_to_ids('、')] = float(0)
                next_token_logits[tokenizer.convert_tokens_to_ids('，')] = float(0)
                next_token_logits[tokenizer.convert_tokens_to_ids('。')] = float(0)
                next_token_logits[tokenizer.convert_tokens_to_ids('；')] = float(0)

            # CH：使用 top_p / top_k 来过滤 next_token_logits
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

            # CH：torch.multinomial() 采样函数
            # num_samples: 采样的次数
            # softmax(): 将所有logits归一化，使其和（sum）为1
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            
            generated_id_tensor = torch.cat((generated_id_tensor, next_token.unsqueeze(0)), dim=1)
    return generated_id_tensor.tolist()[0]

# 3.2 Top-P & Top-K Sampling
# 3.2 Top-P & Top-K 采样函数

def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
) -> Tensor:
    # make sure that the dim of logits tensor is 1
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))  # Safety check

    # the logits of tokens which do not ranked in top-k will be set as -Infinity
    # logits不在top-k的token的logits会被设置为负无穷
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


# 4. Define the main module
# 4. 主模块
def main():

    # 4.1 Title & Sidebar
    # title name of the project
    st.header('【中文文本生成工具箱】')
    # sidebar prompt
    st.sidebar.subheader("功能栏")

    # 4.2 Sidebar Arguments Lists
    # mode switch list
    mode_list = ['普通模式','教学模式']
    # "Teaching Mode" sidebar list
    model_list = ['文章模型','对联模型','歌词模型','古诗模型','散文模型','文言文模型']
    # "Normal Mode" sidebar list
    task_list = ['文章生成','对联生成','歌词生成','古诗生成','散文生成','文言文生成','藏头诗生成']
    poem_subtask_list = ['五言律诗','五言绝句','七言律诗','七言绝句']
    acrostic_sublist = ['五言藏头诗','七言藏头诗']
    set_seed_list = ['不设定','自定义种子', '随机种子']

    # 4.3 Mode Switch
    mode_option = st.sidebar.selectbox(label = '模式选择', options = mode_list)


    # -------------------------------------------- TEACHING MODE -------------------------------------------- #
    if (mode_option == '教学模式'):
        
        # 1. Select Pre-trained Model
        # 1. 选择预训练模型
        model_option = st.sidebar.selectbox(label = '预训练模型选择', options = model_list)
        specific_model = ''
        if (model_option == '文言文模型'):
            specific_model = 'ancient'
            model, tokenizer = setting_model(device,"pretrained_model/ancient", "vocab/ancient/vocab.txt")
        elif (model_option == '文章模型'):
            specific_model = 'article'
            model, tokenizer = setting_model(device,"pretrained_model/article-fast", "vocab/article-fast/vocab.txt")
        elif (model_option == '对联模型'):
            specific_model = 'couplet'
            model, tokenizer = setting_model(device,"pretrained_model/couplet", "vocab/couplet/vocab.txt")
        elif (model_option == '歌词模型'):
            specific_model = 'lyric'
            model, tokenizer = setting_model(device,"pretrained_model/lyric", "vocab/lyric/vocab.txt")
        elif (model_option == '古诗模型'):
            specific_model = 'poem'
            model, tokenizer = setting_model(device,"pretrained_model/poem", "vocab/poem/vocab.txt")
        elif (model_option == '散文模型'):
            specific_model = 'prose'
            model, tokenizer = setting_model(device,"pretrained_model/prose", "vocab/prose/vocab.txt")

        # 2. Hyper-Parameters Sidebar
        # 2. 侧边栏超参数
        repetition_penalty = st.sidebar.number_input("重复处罚率", min_value=0.0, max_value=10.0, value=1.2, step=0.1, help = "用来避免生成文本的无意义重复。该数值越高，重复的文本越少。")
        temperature = st.sidebar.number_input("温度值", min_value=0.0, max_value=10.0, value=1.0, step=0.01, help = "温度是一个用于决策输出结果的值（一般取值是0到1之间）。温度越低，生成的结果越偏向注意力高的词；温度越高，生成的结果会更多样化。")
        top_x = st.sidebar.slider("显示“top-x”的logits", min_value=5, max_value=50, value=20, step=1, help ="显示Logits排名前X的数据显示" )
    
        # 3. Pass the advanced hyper-parameters to "args"
        # 3. 将超参数传递给 "args"
        # 重复处罚率，默认值：1.2
        # 温度值，默认值：1.0
        parser = argparse.ArgumentParser()
        parser.add_argument('--repetition_penalty', default=1.2, type=float, help='重复处罚率')
        parser.add_argument('--temperature', default=1.0, type=float, help='生成文本的温度')
        args = parser.parse_args()


        # 4. Input Box and Prompt
        # 4. 文本输入框及提示信息
        if (specific_model == 'ancient'):
            content = st.text_area(label = "请在这里输入一些文字：", max_chars=512, value = "壬戌之秋，")
        elif (specific_model == 'article'):
            content = st.text_area(label = "请在这里输入一些文字：", max_chars=512, value = "我三月份就该走了，")
        elif (specific_model == 'couplet'):
            content = st.text_area(label = "请在这里输入一些文字：", max_chars=512, value = "爆竹一声除旧岁-")
        elif (specific_model == 'lyric'):
            content = st.text_area(label = "请在这里输入一些文字：", max_chars=512, value = "最美的不是下雨天，是曾与你躲过雨的屋檐。")
        elif (specific_model == 'poem'):
            content = st.text_area(label = "请在这里输入一些文字：", max_chars=512, value = "千山鸟飞绝，")
        elif (specific_model == 'prose'):
            content = st.text_area(label = "请在这里输入一些文字：", max_chars=512, value = "我在雨里站了很久，")


        # 5. Showing Next Token and its Logits
        # 5. 显示 Next Token & Logits 的预测信息
        if st.sidebar.button(label = '预测Next Token'):

            # check the input (cannot be None)
            if (content == ''):
                    st.error("输入不能为空，请写些什么吧。")
                    st.stop()

            # splitting the input
            content_split = tokenizer.tokenize(content)

            # convert the token to "id list"
            content_ids = tokenizer.convert_tokens_to_ids(content_split)
            
            # convert the "id list" to "id tensor"
            content_id_tensor = torch.tensor(content_ids, dtype=torch.long, device=device)
            content_id_tensor = content_id_tensor.unsqueeze(0)
            generated_id_tensor = content_id_tensor
            inputs = {'input_ids': generated_id_tensor[0][:].unsqueeze(0)}

            # model processing
            outputs = model(**inputs)

            # acquire next token logits
            next_token_logits = outputs[0][0, -1, :]

            # apply repetition penalty
            for id in set(generated_id_tensor):
                next_token_logits[id] /= repetition_penalty

            # apply temperature
            next_token_logits = next_token_logits / temperature

            # 统计正数logits的token个数
            # 并且建立正数logits的token字典（所有token按照logits降序排列）
            num_positive = 0
            positive_logits_dict = {}

            for i in range(model.config.vocab_size):
                if (next_token_logits[i]>0):
                    positive_logits_dict[i] = next_token_logits[i]
                    num_positive += 1

            positive_logits_dict = sorted(positive_logits_dict.items(),key=lambda item:item[1], reverse=True)

            # 前X个正数logits的显示
            top_x_logits = []
            top_x_tokens= []
            top_x_ids = []

            for i in range(top_x):
                top_x_logits.append(float(positive_logits_dict[i][1]))
                top_x_tokens.append(tokenizer.decode(positive_logits_dict[i][0]))
                top_x_ids.append(positive_logits_dict[i][0])
            
            # 输出结果
            column_1, column_2 = st.columns(2)
            # column 1 输出 "top-x" logits的 DataFrame
            with column_1:
                top_x_data = {'token':top_x_tokens, 'id':top_x_ids, 'logits':top_x_logits}
                df = pd.DataFrame(top_x_data, index = range(1,top_x+1))

                st.subheader("Next Token Logits 排名")
                st.dataframe(df, height=top_x*32, width=500)

            # column 2 输出 token的概率图
            with column_2:
                st.subheader("Next Token 概率图")
                # 水平条形图（obsolete version）
                # df_barchart = pd.DataFrame(data = top_x_prob, index = top_x_tokens)
                # st.bar_chart(df_barchart, use_container_width=True)
                # 垂直条形图
                top_x_prob = F.softmax(torch.tensor(top_x_logits))
                df_alt = pd.DataFrame({"token":top_x_tokens, "probability": top_x_prob.tolist()})
                vertical = alt.Chart(df_alt).mark_bar().encode(x="probability:Q", y="token:O").properties(height=top_x*28+50)
                st.altair_chart(vertical, use_container_width=True)



    # -------------------------------------------- TEACHING MODE -------------------------------------------- #





    # -------------------------------------------- NORMAL MODE -------------------------------------------- #
    if (mode_option == '普通模式'):
        # 1. Sub-Task Selection
        # 1.1 select a text generation sub-tasks
        nlp_task_option = st.sidebar.selectbox(label = '文本生成任务', options = task_list)
        specific_task = ''
        if (nlp_task_option == '文言文生成'):
            specific_task = 'ancient'
            model, tokenizer = setting_model(device,"pretrained_model/ancient", "vocab/ancient/vocab.txt")
        elif (nlp_task_option == '文章生成'):
            specific_task = 'article'
            model, tokenizer = setting_model(device,"pretrained_model/article-fast", "vocab/article-fast/vocab.txt")
        elif (nlp_task_option == '对联生成'):
            specific_task = 'couplet'
            model, tokenizer = setting_model(device,"pretrained_model/couplet", "vocab/couplet/vocab.txt")
        elif (nlp_task_option == '歌词生成'):
            specific_task = 'lyric'
            model, tokenizer = setting_model(device,"pretrained_model/lyric", "vocab/lyric/vocab.txt")
        elif (nlp_task_option == '古诗生成'):
            specific_task = 'poem'
            model, tokenizer = setting_model(device,"pretrained_model/poem", "vocab/poem/vocab.txt")
        elif (nlp_task_option == '散文生成'):
            specific_task = 'prose'
            model, tokenizer = setting_model(device,"pretrained_model/prose", "vocab/prose/vocab.txt")
        elif (nlp_task_option == '藏头诗生成'):
            specific_task = 'acrostic'
            model, tokenizer = setting_model(device,"pretrained_model/poem", "vocab/poem/vocab.txt")


        # 1.2 setting sub-task arguments
        # 1.2.1 poem generation arguments
        # 古诗生成-子参数
        if (specific_task == 'poem'):
            poem_subtask_option = st.sidebar.selectbox(label = '具体任务参数', options = poem_subtask_list)
            if (poem_subtask_option == '五言律诗'):
                poem_subtask = 'rhyme_5'
                generate_length = 49 # 40(content) + 8(symbols) + 1([CLS])
            elif (poem_subtask_option == '七言律诗'):
                poem_subtask = 'rhyme_7'
                generate_length = 65 # 56(content) + 8(symbols) + 1([CLS])
            elif (poem_subtask_option == '五言绝句'):
                poem_subtask = 'quatrain_5'
                generate_length = 25 # 20(content) + 4(symbols) + 1([CLS])
            elif (poem_subtask_option == '七言绝句'):
                poem_subtask = 'quatrain_7'
                generate_length = 33 # 28(content) + 4(symbols) + 1([CLS]) 

        # 1.2.2 poem generation arguments
        # 藏头诗生成-子参数
        elif (specific_task == 'acrostic'):
            acrostic_sublist_option = st.sidebar.selectbox(label = '藏头诗类型', options = acrostic_sublist)
            if (acrostic_sublist_option == '五言藏头诗'):
                acrostic_mode = "wuyan"
                generate_length = 4 # 每次生成4个字，然后添加标点
            elif (acrostic_sublist_option == '七言藏头诗'):
                acrostic_mode = "qiyan"
                generate_length = 6 # 每次生成4个字，然后添加标点




        # 1.2.3 ancient Chinese generation arguments
        elif (specific_task == 'ancient' or specific_task == 'article' or specific_task == 'lyric' or specific_task == 'prose'):
            generate_length = st.sidebar.number_input(label = '生成文本长度', min_value = 15, max_value = 512, step = 1, value = 100)

        # 2. Other Settings
        # Auto-Save Settings
        save_or_not = st.sidebar.checkbox(label = '自动保存生成结果', value = False, help = '文件保存在当前路径下，名称格式为 "类型-时间.txt"')
        # Do_sample Settings
        do_sample = st.sidebar.checkbox(label = '使用默认解码算法', value = True, help = '开启时，使用"top-p & top-k"算法解码；关闭后，使用贪心算法进行解码，输出结果会变差')

        
        # 3. Seed
        # 3.1 setting seed
        if (do_sample):
            set_seed_choice = st.sidebar.selectbox(label = '是否设定种子', options = set_seed_list, help = '种子是一个决定生成结果的数，若种子不变，生成的结果也不变。因此使用种子可以复现某些结果。注意：使用种子后，生成文本的效果会比较差。')
            if (set_seed_choice == '自定义种子'):
                if_set_seed = 1
            elif (set_seed_choice == '随机种子'):
                if_set_seed = 2
            elif (set_seed_choice == '不设定'):
                if_set_seed = 0
            # 3.2 seed input panel
            if (if_set_seed == 1):
                seed_number = st.sidebar.number_input(label = '种子数值',step = 1,min_value = 0, max_value = 2**32 - 1, help = '请输入整数')
                set_seed(int(seed_number))
            elif (if_set_seed == 2):
                seed_number = random.randint(0, 2**32 - 1)
                set_seed(int(seed_number))


        # 4. Advanced Pipeline Settings
        # 4.1 whether switch on advanced setting
        if (specific_task == 'acrostic'):
            advanced_open = True
            st.sidebar.info("藏头诗模式自动开启高级设置")
        else:
            advanced_open = st.sidebar.checkbox(label = '启用高级设置', value = False, help = "选择古诗模式可生成藏头诗")

        # 4.2 advanced hyper-parameters panel
        if (advanced_open):
            # 解码参数
            repetition_penalty = st.sidebar.number_input("重复处罚率", min_value=0.0, max_value=10.0, value=1.2, step=0.1, help = "用来避免生成文本的无意义重复。该数值越高，重复的文本越少。")
            temperature = st.sidebar.number_input("温度值", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help = "温度是一个用于决策输出结果的值。温度越低，生成的结果越偏向概率高的词；温度越高，生成的结果会更多样化。")
            top_p = st.sidebar.number_input("top_p解码", min_value=0.0, max_value=1.0, value=0.95, step=0.01, help = "解码时，概率累加大于top_p的token会被保留。")
            top_k = st.sidebar.slider("top_k解码", min_value=0, max_value=50, value=40, step=1, help = "解码时，概率累加大于top_k的token会被保留。")
            

        # 5. Pass the advanced hyper-parameters to "args"
        if (advanced_open):
            parser = argparse.ArgumentParser()
            parser.add_argument('--repetition_penalty', default=repetition_penalty, type=float, help='重复处罚率')
            parser.add_argument('--top_k', default=top_k, type=int, help='解码时，保留概率累加大于多少的标记')
            parser.add_argument('--top_p', default=top_p, type=float, help='解码时，保留概率累加大于多少的标记')
            parser.add_argument('--temperature', default=1, type=float, help='生成文本的温度')
            args = parser.parse_args()
        

        # 6. Setting the text generation pipeline
        text_generator = TextGenerationPipeline(model, tokenizer)
        

        # 7. Input Box and Prompt
        if (specific_task == 'ancient'):
            content = st.text_area(label = "请输入一段文言文", max_chars=512, value = "壬戌之秋，")
        elif (specific_task == 'article'):
            content = st.text_area(label = "请输入一句话", max_chars=512, value = "我三月份就该走了，")
        elif (specific_task == 'couplet'):
            content = st.text_area(label = "请在输入末尾添加短横线\"-\"", max_chars=512, value = "爆竹一声除旧岁-")
        elif (specific_task == 'lyric'):
            content = st.text_area(label = "请输入一段歌词", max_chars=512, value = "最美的不是下雨天，是曾与你躲过雨的屋檐。")
        elif (specific_task == 'poem'):
            if (poem_subtask_option == '五言律诗' or poem_subtask_option == '五言绝句'):
                content = st.text_area(label = "请输入五个字，并在末尾添加逗号或句号", max_chars=512, value = "千山鸟飞绝，")
            elif (poem_subtask_option == '七言律诗' or poem_subtask_option == '七言绝句'):
                content = st.text_area(label = "请输入七个字，并在末尾添加逗号或句号", max_chars=512, value = "春江潮水连海平，")      
        elif (specific_task == 'prose'):
            content = st.text_area(label = "请输入一句话", max_chars=512, value = "我在雨里站了很久，")
        elif (specific_task == 'acrostic'):
            content = st.text_area(label = "请输入每一句诗的开头", max_chars=30, value = "春夏秋冬")


        # 8. Additional Operations
        # add special token: [CLS]
        raw_content = content
        content = '[CLS]' + content
        # remove blank symbols
        content.strip()


        # the length of couplet cannot be determined in advance
        if (specific_task == 'couplet'):
            length_of_first_couplet = len(raw_content)-1 # 1: "-"
            if (not advanced_open):
                generate_length = length_of_first_couplet*2+3 # 3([CLS], [SEP], -)
            else:
                generate_length = length_of_first_couplet*2+1
    




    # ----------------------------------------- 处理输入文本 ----------------------------------------- #

    if (mode_option == '普通模式'):
        if st.button(label = "开始生成"):

            # 检查用户输入，不能为空文本
            if (content == ''):
                st.error("输入不能为空，请写些什么吧。")
                st.stop()

            # 检查输入时，不检查添加的 “[CLS]” 标记，因此使用 raw_content
            if (len(raw_content)>generate_length):
                # 不是 “藏头诗模式” & “古诗模式” 时，才检查生成文本的长度
                if (specific_task != "couplet" and specific_task != 'acrostic'):
                    st.error("输入文本超过了生成文本长度了哦，请重新调整。")
                    st.stop()


            ##############################################################################################
            # 开始计算文本生成时间
            process_begin_time = time.time()

            ##############################################################################################
            # NORMAL MODE
            # 未开启高级设置 [0]
            if (not advanced_open):
                
                # 使用默认的 text-generation pipeline 生成文本
                process_result = text_generator(content, max_length = generate_length, do_sample = do_sample)

                # 不启用高级设置时，处理pipeline的多余输出
                process_result = str(process_result)
                process_result = process_result.strip('[')
                process_result = process_result.strip(']')
                process_result = process_result.strip('{')
                process_result = process_result.strip('}')
                process_result = process_result.replace("'generated_text': '[CLS]", "")
                process_result = process_result.replace("\'", "")
                process_result = re.sub('\s','',process_result)


            ##############################################################################################
            # CUSTOMIZED MODE
            # 高级模式，根据超参数定义解码行为
            if (advanced_open == True and specific_task != 'acrostic'):
                process_result = ""

                content_split = tokenizer.tokenize(content)
                content_ids = tokenizer.convert_tokens_to_ids(content_split)
                # times of generation counter

                if (specific_task == 'poem'):
                    generate_length = generate_length-1
                    
                output_ids = decoding(
                    model=model,
                    content_ids=content_ids,
                    length=generate_length - len(raw_content),
                    tokenizer=tokenizer,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    device=device,
                    no_symbol=False)
                    
                text = tokenizer.convert_ids_to_tokens(output_ids)


                # post-processing: remove redundant symbols and special tokens
                for index, token in enumerate(text):
                    if (token == '[MASK]') or (token == '[CLS]') or (token == '[SEP]') or (token == '[UNK]'):
                        text[index] = ''

                process_result = ''.join(text) # text is a list
                process_result = process_result.replace('##', '')
                process_result = process_result.strip()

            ##############################################################################
            # 生成藏头诗，默认开启高级模式
            elif (specific_task == 'acrostic'):
                # check whether user provide input
                if (content == ''):
                    st.error("输入不能为空，请写些什么吧。")
                    st.stop()

                # 准备需要的变量
                process_result = ""

                #对输入预先分词
                # content[0:6] 取 "[CLS]" + 第一个字
                content_split = tokenizer.tokenize(content[0:6])
                content_ids = tokenizer.convert_tokens_to_ids(content_split)

                # 根据输入文本确定循环次数 （循环次数：藏头诗中，“头”的个数）
                # 输入有几个字，就生成几句诗
                # 句的数量 == len(raw_content)

                for i in range(len(raw_content)):
                    output_ids = decoding(
                    model=model,
                    content_ids=content_ids,
                    length=generate_length,
                    tokenizer=tokenizer,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    device=device,
                    no_symbol=True)
                    
                    if (i != (len(raw_content)-1) ): # 将生成的文本变为输入
                        content_ids = output_ids

                        # 添加标点，i为偶数添加逗号；i为奇数添加句号
                        if (i%2 == 0):
                            content_ids.append(tokenizer.convert_tokens_to_ids('，'))
                        else:
                            content_ids.append(tokenizer.convert_tokens_to_ids('。'))

                        next_character_token = list(raw_content[i+1])
                        next_character_id = tokenizer.convert_tokens_to_ids(next_character_token)
                        content_ids.append(next_character_id[0])



                #循环结束，生成完毕，添加最后一个句号，然后去除无用token
                output_ids.append(tokenizer.convert_tokens_to_ids('。'))
                text = tokenizer.convert_ids_to_tokens(output_ids)
                # post-processing: remove redundant symbols and special tokens
                for index, token in enumerate(text):
                    if (token == '[MASK]') or (token == '[CLS]') or (token == '[SEP]') or (token == '[UNK]'):
                        text[index] = ''

                process_result = ''.join(text) # text is a list
                process_result = process_result.replace('##', '')
                process_result = process_result.strip()



            # ------------------------------------------ 额外功能 ------------------------------------------ #

            # 1. 计算生成结果长度
            # acquire the result length
            result_length = len(process_result)

            
            # 2. 计算处理时间
            # calculate the process time
            process_finish_time = time.time()
            process_time = process_finish_time - process_begin_time

            
            # 3. 输出处理用时、处理结果
            # finish and print result
            st.success(" 结果已生成！用时%.3f秒。" %(process_time) )
            st.text_area(label = "输出结果：", value = process_result)


            # 4. 输出实际生成结果长度
            # output real generated length
            if (specific_task == 'poem'): #（古诗生成模式需要提醒理论生成长度）
                if (poem_subtask == 'rhyme_5'):
                    st.text_input(label = "实际生成字数（五言律诗应为48，否则应重新生成）：", value = result_length)
                elif (poem_subtask == 'rhyme_7'):
                    st.text_input(label = "实际生成字数（七言律诗应为64，否则应重新生成）：", value = result_length)
                elif (poem_subtask == 'quatrain_5'):
                    st.text_input(label = "实际生成字数（五言绝句应为24，否则应重新生成）：", value = result_length)
                elif (poem_subtask == 'quatrain_7'):
                    st.text_input(label = "实际生成字数（七言绝句应为32，否则应重新生成）：", value = result_length)
            else: #（其他生成模式不需要提醒理论长度）
                st.text_input(label = "实际生成字数：", value = result_length)


            # 5. 输出种子值
            # Tell the user what the current seed is
            if (if_set_seed == 1 or if_set_seed ==2):
                st.text_input(label = "当前种子数值：", value = seed_number)


            # 6. 是否自动保存
            # whether auto save
            if (save_or_not == True):
                result_save(specific_task, process_result, save=True)
                st.success("文件已自动保存！")


            # 7. 下载生成结果按钮
            # download the generated result
            file_name = result_save(specific_task, process_result, save=False)
            st.download_button (label = "下载生成结果", data = process_result, file_name = file_name)

            # ------------------------------------------ 额外功能 ------------------------------------------ #
    
    # -------------------------------------------- NORMAL MODE -------------------------------------------- #



if __name__ == '__main__':
    main()