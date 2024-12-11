import torch
from tqdm import tqdm

from .utils import ctc_decode, labels_to_string
from .metrics import word_accuracy, character_accuracy, average_levenshtein_distance, character_confusion_matrix

def evaluate(model, dataloader, criterion, label2char,
             max_iter=None, decode_method='beam_search', beam_size=10):
    model.eval()

    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    tot_char_correct = 0
    tot_char_count = 0
    wrong_cases = []

    pbar_total = max_iter if max_iter else len(dataloader)
    pbar = tqdm(total=pbar_total, desc="Evaluate")

    all_preds = []
    all_targets = []

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
                all_preds.append(pred_str)
                all_targets.append(real_str)
                if pred_str == real_str:
                    tot_correct += 1
                else:
                    wrong_cases.append((real_str, pred_str, images[idx].cpu().numpy()))

                # Character accuracy
                tot_char_correct += sum(p == r for p, r in zip(pred_str, real_str))
                tot_char_count += len(real_str)

            pbar.update(1)
        pbar.close()

    word_acc = word_accuracy(all_preds, all_targets)
    char_acc = character_accuracy(all_preds, all_targets)
    avg_edit_distance = average_levenshtein_distance(all_preds, all_targets)
    char_conf_matrix = character_confusion_matrix(all_preds, all_targets, label2char)

    evaluation = {
        'loss': tot_loss / tot_count,
        'word_acc': word_acc,
        'char_acc': char_acc,
        'avg_edit_distance': avg_edit_distance,
        'wrong_cases': wrong_cases,
        'char_conf_matrix': char_conf_matrix
    }
    return evaluation