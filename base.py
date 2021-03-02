import torch
from torch import nn
import torch.nn.functional as f


class ImageClassificationBase(nn.Module):
    def accuracy(self, predictions, targets):
        _, outputs = torch.max(predictions, dim=1)
        return torch.tensor(torch.sum(outputs == targets).item() / len(outputs))

    def training_step(self, batch):
        images, labels = batch
        predictions = self(images)
        loss = f.cross_entropy(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, {'loss': loss.detach(), 'accuracy': accuracy}

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.eval()
        outputs = [self.validation_step(batch) for batch in val_loader]
        return self.epoch_end(outputs)

    @torch.no_grad()
    def predict(self, images):
        self.eval()
        preds = self(images)
        _, outputs = torch.max(preds, dim=1)
        return outputs

    def validation_step(self, batch):
        loss, result = self.training_step(batch)
        return result

    def epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accuracy = [x['accuracy'] for x in outputs]
        epoch_acc = torch.stack(batch_accuracy).mean()
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}

    def epoch_end_log(self, epoch, results):
        print(f'Epoch [{epoch + 1}], train_loss: {results["train_loss"]:.2f} train_accuracy: {results["train_acc"]:.2f} validation_loss: {results["val_loss"]:.2f} val_accuracy: {results["val_acc"]:.2f}')