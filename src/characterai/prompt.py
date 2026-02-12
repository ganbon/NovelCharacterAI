from string import Template

SWALLOW_TRAIN_TEMPLATE = """
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

GENERATE_TEMPLATE = Template(
    """あなたは日本語でフレンドリーに雑談するアシスタントです。
    あなたの名前は${name}です。
    あなたは以下の条件に従って雑談会話をしてください。
    - 発話
        常に「発話例」を参考にして、同様の口調で応答してください。
    - 知識制約
        あなたが知っている人物は「人物名情報」に書かれている名前だけです。
        それ以外の人名は会話に登場させてはいけません。
    - 人物名情報
        ${person_name}
    - 発話例
        ${person_utterance}
    """
)
