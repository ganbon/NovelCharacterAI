from argparse import ArgumentParser
import json


def extract_chracter_name(dialogue, charachter_list):
    name_in_dialogue_list = []
    for utterance in dialogue:
        utterance = utterance["utterance"]
        for character in charachter_list:
            name_list = sorted(character["name"], key=lambda x: len(x), reverse=True)
            for name in name_list:
                if name in utterance:
                    name_in_dialogue_list.append(name)
                    break
    return list(set(name_in_dialogue_list))

def turn_cutting(dialogue_data) -> list[dict]:
    def reset(turn_dialogue, speaker_id, dialogue) -> tuple[list[dict], int, int, int]:
        before_speaker_id = turn_dialogue[-1]["speaker_id"]
        if speaker_id == before_speaker_id:
            turn_dialogue = [dialogue]
            odd_speaker_id, even_speaker_id = -100, speaker_id
            turn_count = 1
        else:
            turn_dialogue = [turn_dialogue[-1], dialogue]
            odd_speaker_id, even_speaker_id = speaker_id, before_speaker_id
            turn_count = 2
        return turn_dialogue, odd_speaker_id, even_speaker_id, turn_count
    
    # def reset(d_list,p_list,person,sentence):
    #     before_person = p_list[-1]
    #     if person == before_person:
    #         d_list, p_list = [sentence], [person]
    #         odd_person,even_person = -100,person
    #         turn_count = 1
    #     else:
    #         d_list, p_list = [d_list[-1],sentence], [before_person,person]
    #         odd_person,even_person = person,before_person
    #         turn_count = 2
    #     return d_list,p_list,odd_person,even_person,turn_count
    
    odd_speaker_id, even_speaker_id = -100,-100
    turn_dialogue_list, turn_dialogue = [], []
    turn_count = 0
    for dialogue in dialogue_data["utterances"]:
        speaker_id = dialogue["speaker_id"]
        if (odd_speaker_id == -100 and even_speaker_id != speaker_id) or (even_speaker_id == -100 and odd_speaker_id != speaker_id):
            turn_dialogue.append(dialogue)
        elif (turn_count%2 and speaker_id == even_speaker_id) or (turn_count%2 == 0 and speaker_id == odd_speaker_id):
            if len(turn_dialogue) >= 2:
                turn_dialogue_list.append(turn_dialogue)
                turn_dialogue = [dialogue]
                odd_speaker_id ,even_speaker_id = -100, speaker_id
                turn_count = 1
                continue
        elif speaker_id == -1:
            if (turn_count%2 == 0 and odd_speaker_id != -1 and speaker_id == even_speaker_id) or \
                (turn_count%2 and even_speaker_id != -1 and speaker_id == odd_speaker_id):
                turn_dialogue.append(dialogue)
            else:
                if len(turn_dialogue) >= 2:
                    turn_dialogue_list.append(turn_dialogue)
                turn_dialogue, odd_speaker_id, even_speaker_id ,turn_count = reset(turn_dialogue, speaker_id, dialogue)
                continue
        else:
            if (turn_count%2 == 0 and speaker_id == even_speaker_id) or \
                (turn_count%2 and speaker_id == odd_speaker_id):
                turn_dialogue.append(dialogue)
            else:
                if len(turn_dialogue) >= 2:
                    turn_dialogue_list.append(turn_dialogue)
                turn_dialogue, odd_speaker_id, even_speaker_id ,turn_count = reset(turn_dialogue, speaker_id, dialogue)
                continue
        if turn_count%2:
            odd_speaker_id = speaker_id
        else:
            even_speaker_id = speaker_id
        turn_count += 1
    if len(turn_dialogue) >= 2:
            turn_dialogue_list.append(turn_dialogue)
    return turn_dialogue_list

def select_response(dialogues, limit_length) -> list[dict]:
    end_index = 0
    for dialogue in reversed(dialogues):
        if len(dialogue["utterance"]) <= limit_length:
            end_index += -1
        else:
            break
    if end_index < 0:
        dialogues = dialogues[:end_index]
    return dialogues

def name_check(dialogues, target_name, person_name):
    if target_name == -1:
        return False
    target_name_list = [name["name"] for name in person_name if name["id"] == target_name][0]
    for i, dialogue in enumerate(reversed(dialogues)):
        utterance = dialogue["utterance"]
        if i % 2 == 0:
            for name in target_name_list:
                if name in utterance:
                    return False
    return True


def main():
    parser = ArgumentParser()
    parser.add_argument("novel_path", type=str)
    parser.add_argument("character_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--dialogue_length", type=int, default=3)
    parser.add_argument("--limit_length", type=int, default=5)
    args = parser.parse_args()
    limit_length, dialogue_length = args.limit_length, args.dialogue_length
    with open(args.novel_path, "r", encoding="utf-8") as f:
        novel_dialogue_corpus = json.load(f)
    with open(args.character_path, "r", encoding="utf-8") as f:
        character_list = json.load(f)
    turn_dialogue_list = []
    for dialogue in novel_dialogue_corpus:
        turn_dialogue_list += turn_cutting(dialogue)
    filtered_turn_dialogue_list = [select_response(dialogue, limit_length) for dialogue in turn_dialogue_list]
    filtered_turn_dialogue_list = [dialogue for dialogue in turn_dialogue_list if len(dialogue) >= dialogue_length]
                    
    id_num = 0
    for dialogue in filtered_turn_dialogue_list:
        target_character_id = dialogue[-1]["speaker_id"]
        if name_check(dialogue, target_character_id, character_list): 
            character_name_in_dialogue = extract_chracter_name(dialogue, character_list)
            if target_character_id != -1:
                with open(args.output_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id": id_num,
                                        "character_id":target_character_id,
                                        "character": dialogue[-1]["speaker_name"],
                                        "utterances": [dial["utterance"] for dial in dialogue],
                                        "character_name_in_dialogue":character_name_in_dialogue}, ensure_ascii=False) + "\n")
                id_num += 1
        copy_dialogue = dialogue.copy()
        target_character_id = copy_dialogue[-2]["speaker_id"]  
        if name_check(copy_dialogue, target_character_id, character_list) and len(copy_dialogue) - 1 >= dialogue_length \
            and len(copy_dialogue[-2]["utterance"]) >= limit_length and target_character_id != -1:
            character_name_in_dialogue = extract_chracter_name(copy_dialogue, character_list)
            copy_dialogue = copy_dialogue[:-1]
            with open(args.output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"id": id_num,
                                    "character_id":target_character_id,
                                    "character": copy_dialogue[-1]["speaker_name"],
                                    "utterances": [dial["utterance"] for dial in copy_dialogue],
                                    "character_name_in_dialogue":character_name_in_dialogue
                                    }, ensure_ascii=False) + "\n")
            id_num += 1


if __name__ == "__main__":
    main()
