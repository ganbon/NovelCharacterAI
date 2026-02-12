import json
import argparse

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from characterai.prompt import GENERATE_TEMPLATE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_utterance_path',type=str)
    parser.add_argument('model_path',type=str)
    parser.add_argument('character_id',type=int)

    args = parser.parse_args()

    sample_utterance_path = args.sample_utterance_path
    model_path = args.model_path
    character_id = args.character_id

    with open(sample_utterance_path) as f:
        tone_sample = [json.loads(line) for line in f]
    name_dict = dict(zip([sample["speaker_id"] for sample in tone_sample],
                        [sample["speaker_name"] for sample in tone_sample]))
    sample_utterance_dict = dict(zip([sample["speaker_id"] for sample in tone_sample],
                                     [sample["sample_utterance"] for sample in tone_sample]))
    name = name_dict[character_id]
    tone_sample = "\n".join(sample_utterance_dict[character_id])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                torch_dtype=torch.float16,
                                                device_map="auto")
    messages = [{"role":"system",
                 "content":GENERATE_TEMPLATE.substitute(name=name,
                                                        person_name=[],
                                                        person_utterance=tone_sample)
                                                        }]
    print(name)
    turn_count = 1
    while(1):
        print(f"{turn_count}対話目")
        # if turn_count == 0:
        #     messages.append({"role":"user",
        #                 "content":"雑談対話を始めてください。"})
        # else:
        text = input("自分：")
        messages.append({"role":"user",
                    "content":text})
        prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
                )
        token_ids = tokenizer.encode(
                prompt, add_special_tokens=False, return_tensors="pt"
                )
        with torch.no_grad():
            output_ids = model.generate(
                token_ids.to(model.device),
                max_new_tokens=200,
                # do_sample=False, 
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                # repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id
            )
        output = tokenizer.decode(
            output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True
        )
        print(output)
        messages.append({"role":"assistant",
                        "content":output})
        turn_count += 1
    
if __name__ == "__main__":
    main()