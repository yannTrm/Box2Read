import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss
from tqdm import tqdm
import wandb

from .utils import ctc_decode, labels_to_string


def evaluate(model, dataloader, criterion, label2char,
             max_iter=None, decode_method='beam_search', beam_size=10):
    model.eval()

    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    wrong_cases = []

    pbar_total = max_iter if max_iter else len(dataloader)
    pbar = tqdm(total=pbar_total, desc="Evaluate")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_iter and i >= max_iter:
                break
            device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

            images, targets, target_lengths = [d.to(device) for d in data]

            logits = model(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size).to(device)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            tot_count += batch_size
            tot_loss += loss.item()
            target_length_counter = 0
            for idx, (pred, target_length) in enumerate(zip(preds, target_lengths)):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length
                real_str = labels_to_string(real, label2char)
                pred_str = labels_to_string(pred, label2char)
                if pred_str == real_str:
                    tot_correct += 1
                else:
                    wrong_cases.append((real_str, pred_str, images[idx].cpu().numpy()))

            pbar.update(1)
        pbar.close()

    evaluation = {
        'loss': tot_loss / tot_count,
        'acc': tot_correct / tot_count,
        'wrong_cases': wrong_cases
    }
    return evaluation


def train_batch(model, data, optimizer, criterion, device):
    model.train()
    images, targets, target_lengths = [d.to(device) for d in data]

    logits = model(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size).to(device)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # Gradient clipping
    optimizer.step()
    return loss.item(), log_probs.detach(), targets, target_lengths

def train_model(model, train_loader, valid_loader, label2char, device, 
                lr=0.001, epochs=10, decode_method='beam_search', beam_size=10,
                criterion=None, optimizer=None, project_name="odometer-reader", run_name="milestone-reader",
                checkpoint=5):
    
    if criterion is None:
        criterion = CTCLoss(reduction='sum', zero_infinity=True).to(device)
    
    if optimizer is None:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    wandb.init(project=project_name, config={
        'lr': lr,
        'epochs': epochs,
        'decode_method': decode_method,
        'beam_size': beam_size
    }, name=run_name)
    
    best_val_accuracy = 0.0
    best_model_state = None

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        model.train()
        tot_train_loss = 0.0
        tot_train_count = 0
        tot_correct = 0

        for train_data in tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False):
            loss, log_probs, targets, target_lengths = train_batch(model, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)
            tot_train_loss += loss
            tot_train_count += train_size

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            target_length_counter = 0
            for idx, (pred, target_length) in enumerate(zip(preds, target_lengths)):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length
                real_str = labels_to_string(real, label2char)
                pred_str = labels_to_string(pred, label2char)
                if pred_str == real_str:
                    tot_correct += 1

        train_loss = tot_train_loss / tot_train_count
        train_accuracy = tot_correct / tot_train_count

        # Validation
        model.eval()
        evaluation_val = evaluate(
            model,
            valid_loader,
            criterion,
            label2char,
            decode_method=decode_method,
            beam_size=beam_size
        )

        val_loss = evaluation_val['loss']
        val_accuracy = evaluation_val['acc']

        # Log wrong cases to wandb
        wrong_cases = evaluation_val['wrong_cases']
        for real, pred, image in wrong_cases:
            wandb.log({
                "wrong_cases": wandb.Image(image, caption=f"Real: {real}, Pred: {pred}")
            }, commit=False)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train/train_loss": train_loss,
            "train/train_accuracy": train_accuracy,
            "val/val_loss": val_loss,
            "val/val_accuracy": val_accuracy
        })

        print(f"Epoch {epoch}: "
              f"train_loss={train_loss}, train_accuracy={train_accuracy}, "
              f"val_loss={val_loss}, val_accuracy={val_accuracy}")

        # Update best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()

        # Save checkpoint every `checkpoint` epochs
        if epoch % checkpoint == 0:
            checkpoint_path = f"./checkpoint_epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)
            os.remove(checkpoint_path)

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save the best model to W&B
    best_model_path = "./best_model.pt"
    torch.save(best_model_state, best_model_path)
    # Create an artifact
    artifact = wandb.Artifact('best_model', type='model')
    artifact.add_file(best_model_path)
    # Log the artifact
    wandb.log_artifact(artifact)
    os.remove(best_model_path) 

    # Finish the W&B run
    wandb.finish()

    return model