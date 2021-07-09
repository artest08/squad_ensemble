from pprint import pprint
import numpy as np
import torch
import collections
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers.data.processors.squad import SquadV2Processor
from datasets import load_metric
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from utils import load

processor = SquadV2Processor()
squad_v2_metric = load_metric("squad_v2")
examples = processor.get_dev_examples("./squad/", filename="dev-v2.0.json")
# best_count = 20
# threshold = 0
# max_answer_length = 30

best_count = 3
threshold = -4
max_answer_length = 20

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


#model_name1 = "twmkn9/distilbert-base-uncased-squad2"
#model_name1 = "deepset/roberta-base-squad2"
#model_name2 = "twmkn9/albert-base-v2-squad2"
#model_name1 = "ktrapeznikov/albert-xlarge-v2-squad-v2"
#model_name2 = "twmkn9/distilroberta-base-squad2"
model_name1 = "twmkn9/bert-base-uncased-squad2"
# model_name2 = 'deepset/roberta-base-squad2'
model_name2 = "deepset/electra-base-squad2"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
model1 = AutoModelForQuestionAnswering.from_pretrained(model_name1)
model1.to(device)
separation_token_id1 = tokenizer1.sep_token_id

tokenizer2 = AutoTokenizer.from_pretrained(model_name2)
model2 = AutoModelForQuestionAnswering.from_pretrained(model_name2)
model2.to(device)
separation_token_id2 = tokenizer2.sep_token_id

# tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
# model = DistilBertForQuestionAnswering.from_pretrained(model_name)
#weight_name = 'distilbert.pth'
#load(weight_name, model)
pad_on_right1 = tokenizer1.padding_side == "right"
pad_on_right2 = tokenizer2.padding_side == "right"


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def extract_logits(model, tokenizer, pad_on_right):
    inputs = tokenizer.encode_plus(example.question_text if pad_on_right else example.context_text,
                                   example.context_text if pad_on_right else example.question_text,
                                   truncation="only_second" if pad_on_right else "only_first",
                                   return_tensors='pt')
    inputs.to(device)
    output = model(**inputs)
    start_logits = output.start_logits
    end_logits = output.end_logits
    return inputs, start_logits, end_logits


for example in examples:

    inputs1, start_logits1, end_logits1 = extract_logits(model1, tokenizer1, pad_on_right1)
    inputs2, start_logits2, end_logits2 = extract_logits(model2, tokenizer2, pad_on_right2)

    # convert the token ids from a tensor to a list
    tokens1 = to_list(inputs1['input_ids'])[0]
    tokens2 = to_list(inputs2['input_ids'])[0]

    start_logits1 = to_list(start_logits1)[0]
    end_logits1 = to_list(end_logits1)[0]

    start_logits2 = to_list(start_logits2)[0]
    end_logits2 = to_list(end_logits2)[0]

    # sort our start and end logits from largest to smallest, keeping track of the index
    start_idx_and_logit1 = sorted(enumerate(start_logits1), key=lambda x: x[1], reverse=True)
    end_idx_and_logit1 = sorted(enumerate(end_logits1), key=lambda x: x[1], reverse=True)

    start_idx_and_logit2 = sorted(enumerate(start_logits2), key=lambda x: x[1], reverse=True)
    end_idx_and_logit2 = sorted(enumerate(end_logits2), key=lambda x: x[1], reverse=True)

    start_indexes1 = [idx for idx, logit in start_idx_and_logit1[:best_count]]
    end_indexes1 = [idx for idx, logit in end_idx_and_logit1[:best_count]]

    start_indexes2 = [idx for idx, logit in start_idx_and_logit2[:best_count]]
    end_indexes2 = [idx for idx, logit in end_idx_and_logit2[:best_count]]

    # question tokens are defined as those between the CLS token (101, at position 0) and first SEP (102) token
    question_indexes1 = [i + 1 for i, token in enumerate(tokens1[1:tokens1.index(separation_token_id1)])]
    question_indexes2 = [i + 1 for i, token in enumerate(tokens2[1:tokens2.index(separation_token_id2)])]

    PrelimPrediction = collections.namedtuple(
        "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit", "token_type"]
    )

    prelim_preds = []
    for start_index in start_indexes1:
        for end_index in end_indexes1:
            # throw out invalid predictions
            if start_index in question_indexes1:
                continue
            if end_index in question_indexes1:
                continue
            if end_index < start_index:
                continue
            if end_index - start_index + 1 > max_answer_length:
                continue
            prelim_preds.append(
                PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=start_logits1[start_index],
                    end_logit=end_logits1[end_index],
                    token_type='1'
                )
            )

    for start_index in start_indexes2:
        for end_index in end_indexes2:
            # throw out invalid predictions
            if start_index in question_indexes2:
                continue
            if end_index in question_indexes2:
                continue
            if end_index < start_index:
                continue
            if end_index - start_index + 1 > max_answer_length:
                continue
            prelim_preds.append(
                PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=start_logits2[start_index],
                    end_logit=end_logits2[end_index],
                    token_type='2'
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
        if len(nbest) >= best_count:
            break

        # loop through predictions according to their start index
        if pred.start_index > 0:  # non-null answers have start_index > 0

            if pred.token_type == '1':
                text = tokenizer1.convert_tokens_to_string(
                    tokenizer1.convert_ids_to_tokens(
                        tokens1[pred.start_index:pred.end_index + 1]
                    )
                )
            elif pred.token_type == '2':
                text = tokenizer2.convert_tokens_to_string(
                    tokenizer2.convert_ids_to_tokens(
                        tokens2[pred.start_index:pred.end_index + 1]
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
    score_null1 = start_logits1[0] + end_logits1[0]
    score_null2 = start_logits2[0] + end_logits2[0]
    if score_null1 > score_null2:
        start_logit = start_logits1[0]
        end_logit = end_logits1[0]
        score_null = score_null1
    else:
        start_logit = start_logits2[0]
        end_logit = end_logits2[0]
        score_null = score_null2
    nbest.append(BestPrediction(text="", start_logit=start_logit, end_logit=end_logit))
    # compute the null score as the sum of the [CLS] token logits

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
