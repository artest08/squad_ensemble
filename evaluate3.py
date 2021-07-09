from pprint import pprint
import numpy as np
import torch
import collections
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers.data.processors.squad import SquadV2Processor

from datasets import load_metric

from utils import load

processor = SquadV2Processor()
squad_v2_metric = load_metric("squad_v2")
examples = processor.get_dev_examples("./squad/", filename="dev-v2.0.json")
best_count = 3
threshold = -4
max_answer_length = 20


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def get_gold_answers(example):
    """helper function that retrieves all possible true answers from a squad2.0 example"""

    gold_answers = [answer["text"] for answer in example.answers if answer["text"]]
    gold_starts = [answer["answer_start"] for answer in example.answers if answer["text"]]
    # if gold_answers doesn't exist it's because this is a negative example -
    # the only correct answer is an empty string
    if not gold_answers:
        gold_answers = [""]
        gold_starts = []

    return gold_answers, gold_starts

model_name = 'distilbert-base-uncased'
# model_name = 'albert-base-v2'
#model_name = "twmkn9/bert-base-uncased-squad2"
#model_name = "twmkn9/distilroberta-base-squad2"
# model_name = "twmkn9/distilbert-base-uncased-squad2"
#model_name = "twmkn9/albert-base-v2-squad2"
#model_name = "ktrapeznikov/albert-xlarge-v2-squad-v2"
model_name = "deepset/roberta-base-squad2"
model_name = "deepset/electra-base-squad2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
separation_token_id = tokenizer.sep_token_id


#weight_name = 'distilbert.pth_4'
#weight_name = 'albert_2.pth'
#load(weight_name, model)
pad_on_right = tokenizer.padding_side == "right"



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
for example in examples:
    inputs = tokenizer.encode_plus(example.question_text if pad_on_right else example.context_text,
                                   example.context_text if pad_on_right else example.question_text,
                                   truncation="only_second" if pad_on_right else "only_first",
                                   return_tensors='pt')

    inputs.to(device)
    output = model(**inputs)
    start_logits = output.start_logits
    end_logits = output.end_logits

    start_logits = to_list(start_logits)[0]
    end_logits = to_list(end_logits)[0]

    # sort our start and end logits from largest to smallest, keeping track of the index
    start_idx_and_logit = sorted(enumerate(start_logits), key=lambda x: x[1], reverse=True)
    end_idx_and_logit = sorted(enumerate(end_logits), key=lambda x: x[1], reverse=True)

    start_indexes = [idx for idx, logit in start_idx_and_logit[:best_count]]
    end_indexes = [idx for idx, logit in end_idx_and_logit[:best_count]]

    # convert the token ids from a tensor to a list
    tokens = to_list(inputs['input_ids'])[0]

    # question tokens are defined as those between the CLS token (101, at position 0) and first SEP (102) token
    question_indexes = [i + 1 for i, token in enumerate(tokens[1:tokens.index(separation_token_id)])]

    PrelimPrediction = collections.namedtuple(
        "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
    )

    prelim_preds = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            # throw out invalid predictions
            if start_index in question_indexes:
                continue
            if end_index in question_indexes:
                continue
            if end_index < start_index:
                continue
            if end_index - start_index + 1 > max_answer_length:
                continue
            prelim_preds.append(
                PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=start_logits[start_index],
                    end_logit=end_logits[end_index]
                )
            )

    prelim_preds = sorted(prelim_preds, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

    BestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "BestPrediction", ["text", "start_logit", "end_logit"]
    )

    nbest = []
    seen_predictions = []
    for pred in prelim_preds:

        # for now we only care about the top 5 best predictions
        if len(nbest) >= 1:
            break

        # loop through predictions according to their start index
        if pred.start_index > 0:  # non-null answers have start_index > 0

            text = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(
                    tokens[pred.start_index:pred.end_index + 1]
                )
            )
            # clean whitespace
            text = text.strip()
            text = " ".join(text.split())

            if text in seen_predictions:
                continue

            # flag this text as being seen -- if we see it again, don't add it to the nbest list
            seen_predictions.append(text)

            # add this text prediction to a pruned list of the top 5 best predictions
            nbest.append(BestPrediction(text=text, start_logit=pred.start_logit, end_logit=pred.end_logit))

    # and don't forget -- include the null answer!
    nbest.append(BestPrediction(text="", start_logit=start_logits[0], end_logit=end_logits[0]))
    # compute the null score as the sum of the [CLS] token logits
    score_null = start_logits[0] + end_logits[0]

    # compute the difference between the null score and the best non-null score
    score_diff = score_null - nbest[0].start_logit - nbest[0].end_logit

    if score_diff > threshold:
        predicted_answer = nbest[-1].text
    else:
        predicted_answer = nbest[0].text

    gold_answers, gold_starts = get_gold_answers(example)
    predictions = [{'prediction_text': predicted_answer, 'id': example.qas_id, 'no_answer_probability': 0.0}]

    references = [{'answers': {'answer_start': gold_starts, 'text': gold_answers}, 'id': example.qas_id}]

    squad_v2_metric.add_batch(predictions=predictions, references=references)
results = squad_v2_metric.compute()
print(results)
