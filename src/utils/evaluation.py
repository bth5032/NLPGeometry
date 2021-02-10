# External imports
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, fbeta_score, confusion_matrix)


def test_epoch(model, test_loader, device):
    '''
    Evaluates a model on a test set according to specified metrics.

    Parameters
    ----------
    model: nn.Module
        A torch model.

    test_loader: torch.utils.data.DataLoader child.
        A dataloader containing the data on which the model will be tested.

    device: torch.device
        Device on which to do the evaluation.
    '''
    model.eval()

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch_index, (data, target) in enumerate(test_loader):

            # Move to device
            data, target = data.to(device), target.to(device)

            # Evaluate model on batch
            output = model(data)

            # Store outputs for global metric calculation
            all_outputs.append(output)

            # Store true labels, for same reason
            all_targets.append(target)

    # Convert outputs to numpy array
    all_outputs = torch.cat(all_outputs, dim=0)
    all_outputs = all_outputs.reshape(len(all_outputs), -1).cpu()

    all_targets = torch.cat(all_targets, dim=0).cpu()

    # Multi-class Score based metrics
    try:
        roc_auc = roc_auc_score(all_targets,
                                nn.Softmax(dim=1)(all_outputs),
                                multi_class='ovr')
    except ValueError:
        roc_auc = 0

    # Convert outputs to labels
    _, all_output_labels = all_outputs.max(dim=1)

    # Label based metrics
    accuracy = accuracy_score(all_targets, all_output_labels)
    precision = precision_score(all_targets,
                                all_output_labels,
                                average=None)
    recall = recall_score(all_targets,
                          all_output_labels,
                          average=None)
    f_beta = fbeta_score(all_targets,
                         all_output_labels,
                         beta=1,
                         average=None)

    conf_matrix = confusion_matrix(all_targets,
                                   all_output_labels,
                                   labels=np.unique(all_targets))

    print("Test Set Evaluation:",
          f"ROC AUC: {roc_auc:.2f}\n",
          f"Accuracy: {accuracy:.2f}\n",
          f"Precision: {precision}\n",
          f"Recall: {recall}\n",
          f"F_beta: {f_beta}\n",
          f"Confusion Matrix: {conf_matrix}\n",
          sep='')

    return None
