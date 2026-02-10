from argparse import ArgumentParser
import json

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from characterai.tone_embedding import encode_line

def main():
    parser = ArgumentParser()
    parser.add_argument("novel_path", type=str)
    parser.add_argument("character_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--sample_num", type=int, default=5)
    parser.add_argument("--limit_length", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    sample_num, limit_length, threshold = args.sample_num, args.limit_length, args.threshold 
    tone_model = SentenceTransformer("ganbon/novel-sentence-bert-base-tone-embedding")
    novel_data = pd.read_csv(args.novel_path)

    sample_character_utterance_list = []
    with open(args.character_path, "r", encoding="utf-8") as f:
        character_list = json.load(f)
    novel_person_name_list = []
    for character in character_list:
        novel_person_name_list += character["name"]
    line_data = novel_data[["sentence","speaker"]][novel_data["speaker"] != -1]
    line_data["vector"] = encode_line(tone_model, line_data["sentence"].tolist())
    for speaker in set(line_data["speaker"].tolist()):
        target_sentence = []
        if speaker != -1:
            speaker_data = line_data[line_data["speaker"] == speaker]
            vectors = speaker_data["vector"].tolist()
            mean_vector = np.mean(vectors, axis=0)  
            distance = cosine_similarity(vectors,[mean_vector])
            for dis,sentence in zip(distance, speaker_data["sentence"]):
                sentence = sentence[1:-1]
                flg = False
                if dis >= threshold and len(sentence[1:-1]) >= limit_length:
                    for name in novel_person_name_list:
                        if name in sentence:
                            flg = True
                    if not flg:
                        target_sentence.append([dis, sentence])
            target_sentence = sorted(target_sentence,key=lambda x:x[0],reverse=True) 
            target_sentence_list = [sentence for dis,sentence in target_sentence]
            if len(target_sentence_list) >= sample_num:
                sample_character_utterance_list.append({
                    "speaker_id":speaker,
                    "speaker_name": [character["name"][0] for character in character_list if character["id"] == speaker][0],
                    "sample_utterance": sorted(set(target_sentence_list),key=target_sentence_list.index)[:sample_num]
                })
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for sample_utterance in sample_character_utterance_list:
            f.write(json.dumps(sample_utterance, ensure_ascii=False) + "\n")
        

if __name__ == "__main__":
    main()
    