import os.path
from transformers import DistilBertTokenizerFast
import torch
from transformers import DistilBertForQuestionAnswering
from torch.utils.data import DataLoader
from transformers import AdamW
from datasets import load_metric
from utils import save, train_loop, val_loop
import wandb
from squad.dataset import SquadDataset

# train = load_dataset("squad_v2", split="train")

# 1. Start a W&B run
wandb.init(project='is784_project', entity='esat')

# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 5e-5
config.batch_size = 4
config.model_name = 'distilbert-base-uncased'
config.save_name = "distilbert.pth"


tokenizer = DistilBertTokenizerFast.from_pretrained(config.model_name)
model = DistilBertForQuestionAnswering.from_pretrained(config.model_name)
optimizerADAMW = AdamW(model.parameters(), lr=config.learning_rate)
metric = load_metric("squad_v2")

train_dataset = SquadDataset('squad/train-v2.0.json', tokenizer)
val_dataset = SquadDataset('squad/dev-v2.0.json', tokenizer)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)


best_validation_loss = 100
for epoch in range(20):
    mean_loss_training = train_loop(train_loader, model, optimizerADAMW)
    mean_loss_validation = val_loop(val_loader, model)
    wandb.log({"training_loss": mean_loss_training, "val_loss": mean_loss_validation})
    if mean_loss_validation < best_validation_loss:
        best_validation_loss = mean_loss_validation
        save(model, optimizerADAMW, config.save_name, epoch, best_validation_loss)
# model.eval()
# for epoch in range(1):
#     for batch in train_loader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         start_positions = batch['start_positions'].to(device)
#         end_positions = batch['end_positions'].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
#         predicted_start_pos = torch.argmax(outputs.start_logits)
#         predicted_end_pos = torch.argmax(outputs.end_logits)
#         predicted_output = list_to_strings(tokenizer.convert_ids_to_tokens(input_ids[0])[predicted_start_pos:predicted_end_pos])
#         references = [{'answers': {'answer_start': [batch["answers"]["answer_start"][0].item()],
#                                    'text': batch["answers"]["text"][0]}, 'id': batch["ids"][0]}]
#         predictions = [{'prediction_text': predicted_output, 'id': batch["ids"][0], 'no_answer_probability': 0.0}]
#         metric.add_batch(predictions=predictions, references=references)
# results = metric.compute()
# print(results)
