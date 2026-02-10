import torch
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import LoraConfig
from trl import SFTTrainer ,SFTConfig
import argparse

# 参考URL：https://wandb.ai/npaka3/qlora-zundamon/reports/Elyza-7B-QLoRA---Vmlldzo2MDk3MzYw
# 参考URL:https://note.com/npaka/n/n315c0bdbbf00

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path')
parser.add_argument('sample_utterance_path')
parser.add_argument('save_dir')
parser.add_argument('model_name')
parser.add_argument('name')

args = parser.parse_args()

dataset_path = args.dataset_path
sample_utterance_path = args.sample_utterance_path
save_dir = args.save_dir
model_name = args.model_name

R = 64
STEPS = 20000
SAVE_STEP = 5000

if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 量子化の有効化
    bnb_4bit_quant_type="nf4",  # 量子化種別 (fp4 or nf4)
    bnb_4bit_compute_dtype=torch_dtype,  # 量子化のdtype (float16 or bfloat16)
    bnb_4bit_use_double_quant=True,
)

# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    model_name,  # モデル名
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation = "eager"
)

model.config.use_cache = False  # キャッシュ (学習時はFalse)
model.config.pretraining_tp = 1  # 事前学習で使用したテンソル並列ランク

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # fp16でのオーバーフロー問題対策
chat_tempalte_path = "./dataset/chat_template.jinja"
# tokenizer.chat_template = """
# {% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>

# '+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>

# ' }}{{ content }}{% endgeneration %}{% endif %}
# """
tokenizer.chat_template = """
{%- set loop_messages = messages -%}
{%- for message in loop_messages -%}
    {% if loop.index0 == 0 %}
        {{- bos_token -}}
    {% endif %}
    {%- if message['role'] == 'assistant' %}
        {%- generation -%}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' -}}
        {%- endgeneration -%}
    {%- else -%}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
{%- endif -%}
"""
def set_history(history_list):
    turn_num = len(history_list)
    if turn_num%2 != 0:
        history_messages =  [{'role': "assistant",'content': history} 
                            if h%2== 0 else {'role': "user",'content': history}
                            for h,history in enumerate(history_list)]
    else:
        history_messages = [{'role': "user",'content': history} 
                            if h%2== 0 else {'role': "assistant",'content': history}
                            for h,history in enumerate(history_list)]
    return history_messages

def generate_metadata_prompt(example):
    meta_data = example["unique_word"]
    name = example["name"]
    history_list = set_history(example['sentence'])
    person_utterance = "\n".join(example["tone_sample"])
    
    person_name = '\n'.join([key for key,val in meta_data.items() if val == "Person" and val not in name])
    messages = [
        {
            'role': "system",
            'content': f"""あなたは日本語でフレンドリーに雑談するアシスタントです。
あなたの名前は{name}です。
あなたは以下の条件に従って雑談会話をしてください。
- 発話
    常に「発話例」を参考にして、同様の口調で応答してください。
- 知識制約
    あなたが知っている人物は「人物名情報」に書かれている名前だけです。
    それ以外の人名は会話に登場させてはいけません。
- 人物名情報
    {person_name}
- 発話例
    {person_utterance}
"""
        }
    ]
    messages += history_list
    return messages
    # return tokenizer.apply_chat_template(messages, tokenize=False)


# text列の追加
def add_text(example):
    example["messages"] = generate_metadata_prompt(example)
    for key in ["sentence", "tone_sample","person",'novel_id',"dialogue_num","unique_word","name"]:
        del example[key]
    return example

dataset = load_dataset("json",data_files = dataset_path, split="train")

dataset = dataset.map(add_text)
dataset = dataset.shuffle()

peft_config = LoraConfig(
    r=R,  # LoRAアテンションの次元
    lora_alpha=16,  # LoRAスケーリングのAlphaパラメータ
    lora_dropout=0.05,  # LoRA レイヤーのドロップアウト確率
    bias="none",  # LoRAのバイアス種別 ("none","all", "lora_only")
    task_type="CAUSAL_LM",  # タスク種別
    target_modules=['up_proj', 'down_proj', 'gate_proj',
                     'k_proj', 'q_proj', 'v_proj', 'o_proj'],
)

# 学習パラメータ
training_arguments = SFTConfig(
    output_dir=save_dir,  # 出力ディレクトリ
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False, 
    max_steps=STEPS,  # 学習ステップ数
    per_device_train_batch_size=1,  # 学習用のGPUあたりのバッチサイズ
    gradient_accumulation_steps=4,  # 勾配を蓄積するための更新ステップの数
    optim="paged_adamw_32bit",  # オプティマイザ
    learning_rate=1e-5,  # 初期学習率
    lr_scheduler_type="cosine",  # 学習率スケジュール
    dataset_text_field="messages",
    max_grad_norm=0.3,  # 最大法線勾配 (勾配クリッピング)
    warmup_ratio=0.03,  # 線形ウォームアップのステップ比率 (0から学習率まで)
    weight_decay=0.001,  # bias/LayerNormウェイトを除く全レイヤーに適用するウェイト減衰
    save_steps=SAVE_STEP,  # 何ステップ毎にチェックポイントを保存するか
    logging_steps=25,  # 何ステップ毎にログを記録するか
    group_by_length=True,  # シーケンスを同じ長さのバッチにグループ化 (メモリ節約)
    assistant_only_loss=True,
    report_to="wandb"  # レポート
)

# SFTパラメータ
trainer = SFTTrainer(
    model=model,  # モデル 
    train_dataset=dataset,  # データセット
    peft_config=peft_config,  # PEFTパラメータ
    args=training_arguments,  # 学習パラメータ
    processing_class=tokenizer, # トークナイザー 
)

# モデルの学習
trainer.train()
