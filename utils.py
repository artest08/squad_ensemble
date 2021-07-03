import torch
import os
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def save(model_to_save, optimizer, save_name, epoch_num, best_loss):
    save_location = os.path.join("./model_files", save_name)
    torch.save({
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch_num,
        'best_loss': best_loss
    }, save_location)


def load(load_name, model):
    load_location = os.path.join("./model_files", load_name)
    checkpoint = torch.load(load_location, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])


def model_outputs(model_to_out, batch):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    start_positions = batch['start_positions'].to(device)
    end_positions = batch['end_positions'].to(device)
    outputs = model_to_out(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                           end_positions=end_positions)
    return outputs


def train_loop(loader, model_in_loop, optimizer):
    loss_list = []
    model_in_loop.train()
    for batch in loader:
        optimizer.zero_grad()
        outputs = model_outputs(model_in_loop, batch)
        loss = outputs[0]
        loss_list.append(loss.item())
        # metric.add_batch(predictions=outputs, references=references)
        loss.backward()
        optimizer.step()
    mean_loss = np.mean(loss_list)
    print(f'mean epoch loss training: {mean_loss}')
    return mean_loss


def val_loop(loader, model_in_loop):
    loss_list = []
    model_in_loop.eval()
    for batch in loader:
        outputs = model_outputs(model_in_loop, batch)
        loss = outputs[0]
        loss_list.append(loss.item())
        # metric.add_batch(predictions=outputs, references=references)
    mean_loss = np.mean(loss_list)
    print(f'mean epoch loss validation: {mean_loss}')
    return mean_loss


def list_to_strings(s):
    str1 = " "
    return str1.join(s)
