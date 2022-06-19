import numpy as np
import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score

from tqdm import tqdm
from copy import deepcopy

try:
    from utils import softmax
except:
    def softmax(pred):
        return torch.exp(pred) / torch.sum(torch.exp(pred))


class Trainer:
    '''This trainer is only for classification'''
    
    def __init__(self, model, optimizer, loss_func, num_classes, device, scheduler=None, path_save=None):
        self.model = model
        self.optimizer = optimizer 
        self.loss_func = loss_func
        self.num_classes = num_classes

        self.device = device
        self.scheduler = scheduler

        self.save_model = {'path': path_save, 'smallest_loss': np.inf}
        self.history = {'Train': {'loss': [], 'accuracy': [], 'precision': [], 'recall': []},
                        'Valid': {'loss': [], 'accuracy': [], 'precision': [], 'recall': []}}
        self.last_epoch = False


    def train(self, train_loader, valid_loader, epochs):
        epoch = 1
        if self.history['Train']['loss']:
            epochs += len(self.history['Train']['loss'])
            epoch += len(self.history['Train']['loss'])

        for epoch in range(epoch, epochs + 1):
            self.model.train()
            for_metrics = {'true': [], 'pred': [], 'loss': []}

            loop = tqdm(train_loader)
            loop.set_description(f'[{epoch}/{epochs}][train]')

            for images, targets in loop:
                torch.cuda.memory.empty_cache()
                images, targets = images.to(self.device), targets.to(self.device)

                preds = self.model(images)
                loss = self.loss_func(preds, targets)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                _, preds = torch.max(preds, 1)
                for_metrics['true'].extend(targets.cpu())
                for_metrics['pred'].extend(preds.cpu())
                for_metrics['loss'].append(loss.cpu().item())

                loop.set_postfix(loss=loss.item())
                
            self.compute_metrics(for_metrics, 'Train', len(train_loader))
            
            if self.scheduler:
                self.scheduler.step(self.history['Train']['loss'][-1])

            if epoch == epochs:
                self.last_epoch = True
            
            self.validate(valid_loader, f'[{epoch}/{epochs}][valid]')
        
    
    def compute_metrics(self, for_metrics, mode, len_loader):
        mean_loss = sum(for_metrics['loss']) / len_loader
        mean_accuracy = accuracy_score(for_metrics['true'], for_metrics['pred'])
        precision = precision_score(for_metrics['true'], for_metrics['pred'],
                                    average='binary' if self.num_classes == 2 else 'micro')
        recall = recall_score(for_metrics['true'], for_metrics['pred'],
                                    average='binary' if self.num_classes == 2 else 'micro')

        self.history[mode]['loss'].append(mean_loss)
        self.history[mode]['accuracy'].append(mean_accuracy)
        self.history[mode]['precision'].append(precision)
        self.history[mode]['recall'].append(recall)


    def validate(self, valid_loader, info_tqdm):
        self.model.eval()
        with torch.no_grad():
            for_metrics = {'true': [], 'pred': [], 'loss': []}

            loop = tqdm(valid_loader)
            loop.set_description(info_tqdm)

            for images, targets in loop:
                torch.cuda.memory.empty_cache()
                images, targets = images.to(self.device), targets.to(self.device)

                preds = self.model(images)
                loss = self.loss_func(preds, targets)

                _, preds = torch.max(preds, 1)
                for_metrics['true'].extend(targets.cpu())
                for_metrics['pred'].extend(preds.cpu())
                for_metrics['loss'].append(loss.cpu().item())

                loop.set_postfix(loss=loss.item())
            
            self.compute_metrics(for_metrics, 'Valid', len(valid_loader))

            if self.save_model['path']:
                if self.history['Valid']['loss'][-1] < self.save_model['smallest_loss']:
                    torch.save({'model': deepcopy(self.model),
                                'image_sizes': images.shape[-2:]}, self.save_model['path'])
                    self.save_model['smallest_loss'] = self.history['Valid']['loss'][-1]
                
            self.print_output(loop)

    
    def print_output(self, loop):
        if self.last_epoch: loop.write('')
        loop.write(12 * '_' + 42 * ' ' + 20 * '_')

        for mode in ['Train', 'Valid']:
            all_text = ''
            templates = [format(f'      {mode}' + ' | accuracy: {}'), format('   | precision: {} |\n'),
                         format('\t    | loss: {}'), format('\t | recall: {}    |')]

            for idx, score in enumerate(['accuracy', 'precision', 'loss', 'recall']):
                txt = str(round(self.history[mode][score][-1], 4)).ljust(6, '0')
                all_text += templates[idx].format(txt)

            all_text += '\n\t      ' + '-' * 38
            loop.write(all_text)

        loop.write('_' * 74)


def test_model(model, test_loader, device):
    '''Test model. Return List[Tuple[image, pred, target, probs]]'''
    
    pred_tuples = []
    corrects_pred = 0

    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            torch.cuda.empty_cache()
            images = images.to(device)

            preds = model(images).cpu()
            probs = [softmax(pred) for pred in preds]
            _, preds = torch.max(preds, 1)
            corrects_pred += torch.sum(preds == targets).item()

            for idx in range(len(images)):
                pred_tuples.append((images[idx].cpu(), preds[idx],
                                    targets[idx], probs[idx]))
    
    print('\naccuracy:', corrects_pred / len(test_loader.dataset))
    return pred_tuples