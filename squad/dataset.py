import torch
from pathlib import Path
import json


def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    ids = []
    is_impossibles = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']  # .lower()
            for qa in passage['qas']:
                question = qa['question']  # .lower()
                id_q = qa['id']
                is_impossible = qa['is_impossible']
                # if is_impossible:
                #     print("sdfsdf")
                if len(qa['answers']) != 0:
                    for answer in qa['answers']:
                        contexts.append(context)
                        questions.append(question)
                        # answer["text"] = answer["text"].lower()
                        answers.append(answer)
                        ids.append(id_q)
                        # Su anda bu kisim hic eklenmiyor
                        is_impossibles.append(is_impossible)
                else:
                    contexts.append(context)
                    questions.append(question)
                    answers.append({'text': '', 'answer_start': 0})
                    ids.append(id_q)
                    # Su anda bu kisim hic eklenmiyor
                    is_impossibles.append(is_impossible)

    return contexts, questions, answers, is_impossibles, ids


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx - 1:end_idx - 1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2:end_idx - 2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2  # When the gold label is off by two characters


def add_token_positions(encodings, answers, model_max_length):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        if answers[i]['text'] == '':
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = model_max_length
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


def prepare_features(tokenizer, questions, contexts, all_answers):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    pad_on_right = tokenizer.padding_side == "right"
    tokenized_examples = tokenizer(questions if pad_on_right else contexts,
                                   contexts if pad_on_right else questions,
                                   truncation="only_second" if pad_on_right else "only_first",
                                   max_length=384,
                                   stride=128,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding="max_length")
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = all_answers[sample_index]
        # If no answers are given, set the cls_index as answer.
        if answers["answer_start"] == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"]
            end_char = start_char + len(answers["text"])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def old_prepare_features(tokenizer, questions, contexts, all_answers):
    add_end_idx(all_answers, contexts)
    pad_on_right = tokenizer.padding_side == "right"
    encodings = tokenizer(questions if pad_on_right else contexts,
                          contexts if pad_on_right else questions,
                          truncation="only_second" if pad_on_right else "only_first",
                          max_length=384,
                          stride=128,
                          return_overflowing_tokens=True,
                          return_offsets_mapping=True,
                          padding="max_length")
    add_token_positions(encodings, all_answers, tokenizer.model_max_length)
    return encodings


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file, tokenizer):
        contexts, questions, answers, is_impossibles, ids = read_squad(dataset_file)
        self.encodings = prepare_features(tokenizer, questions, contexts, answers)
        self.dataset_dict = {"contexts": contexts, "questions": questions,
                             "answers": answers, "is_impossibles": is_impossibles, "ids": ids}

    def __getitem__(self, idx):
        return_dict = {}
        return_dict.update({key: torch.tensor(val[idx]) for key, val in self.encodings.items()})
        #return_dict.update({key: val[idx] for key, val in self.dataset_dict.items()})
        return return_dict

    def __len__(self):
        return len(self.encodings.input_ids)
