import os.path
import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AlbertTokenizerFast
from transformers import AlbertForQuestionAnswering
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from torch.utils.data import DataLoader
from transformers import AdamW
from datasets import load_metric
from utils import save, train_loop, val_loop
import wandb
from transformers import get_linear_schedule_with_warmup
from squad.dataset import SquadDataset

# train = load_dataset("squad_v2", split="train")

# 1. Start a W&B run
wandb.init(project='is784_project', entity='esat')
# 2. Save model inputs and hyperparameters
config = wandb.config
config.epoch_number = 3
config.learning_rate = 5e-5
config.batch_size = 5
config.warmup_steps = 100
config.transformer = 'bert'
if config.transformer == 'distilbert':
    config.tokenizer_name = 'distilbert-base-uncased'
    config.model_name = 'distilbert-base-uncased'
elif config.transformer == 'albert':
    config.tokenizer_name = 'albert-base-v2'
    config.model_name = 'albert-base-v2'
elif config.transformer == 'bert':
    config.tokenizer_name = 'bert-base-uncased'
    config.model_name = 'bert-base-uncased'
elif config.transformer == 'albert-distilbert':
    config.tokenizer_name = 'albert-base-v2'
    config.model_name2 = 'distilbert-base-uncased'
config.save_name = config.transformer

tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
model = AutoModelForQuestionAnswering.from_pretrained(config.model_name)


optimizerADAMW = AdamW(model.parameters(), lr=config.learning_rate)
metric = load_metric("squad_v2")

train_dataset = SquadDataset('squad/train-v2.0.json', tokenizer)
val_dataset = SquadDataset('squad/dev-v2.0.json', tokenizer)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

t_total = len(train_loader) // config.epoch_number
scheduler = get_linear_schedule_with_warmup(
    optimizerADAMW, num_warmup_steps=config.warmup_steps, num_training_steps=t_total)




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)




best_validation_loss = 100
for epoch in range(config.epoch_number):
    mean_loss_training = train_loop(train_loader, model, optimizerADAMW, scheduler)
    mean_loss_validation = val_loop(val_loader, model)
    wandb.log({"training_loss": mean_loss_training, "val_loss": mean_loss_validation})
    best_validation_loss = mean_loss_validation
    save(model, optimizerADAMW, config.save_name + f'_{epoch}.pth', epoch, best_validation_loss)
    if mean_loss_validation < best_validation_loss:
        best_validation_loss = mean_loss_validation
        save(model, optimizerADAMW, config.save_name + '.pth', epoch, best_validation_loss)

