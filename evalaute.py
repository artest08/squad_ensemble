import os.path

from transformers import DistilBertTokenizerFast
import torch
from transformers import DistilBertForQuestionAnswering
from squad.dataset import SquadDataset
from torch.utils.data import DataLoader
import os
from utils import list_to_strings



model_name = 'distilbert-base-uncased'
weight_name = 'distilbert.pth'


tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)
load(weight_name, model)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

val_dataset = SquadDataset('squad/dev-v2.0.json', tokenizer)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

model.eval()
for epoch in range(1):
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        predicted_start_pos = torch.argmax(outputs.start_logits)
        predicted_end_pos = torch.argmax(outputs.end_logits)
        predicted_output = list_to_strings(tokenizer.convert_ids_to_tokens(input_ids[0])[predicted_start_pos:predicted_end_pos])
        references = [{'answers': {'answer_start': [batch["answers"]["answer_start"][0].item()],
                                   'text': batch["answers"]["text"][0]}, 'id': batch["ids"][0]}]
        predictions = [{'prediction_text': predicted_output, 'id': batch["ids"][0], 'no_answer_probability': 0.0}]
        metric.add_batch(predictions=predictions, references=references)
results = metric.compute()
print(results)
