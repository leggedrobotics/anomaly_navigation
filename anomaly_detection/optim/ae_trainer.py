from anomaly_detection.base.base_trainer import BaseTrainer
from anomaly_detection.base.base_dataset import BaseADDataset
from anomaly_detection.base.base_net import BaseNet
from anomaly_detection.utils.eval_functions import computeMaxYoudensIndex
from sklearn.metrics import roc_auc_score, roc_curve

import logging
import time
import torch
import torch.optim as optim
import numpy as np


class AETrainer(BaseTrainer):

    def __init__(self, writer, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader, writer)

    def train(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        best_auc = 0

        # Training
        logger.info('Starting pretraining...')
        start_time = time.time()
        ae_net.train()
        for epoch in range(self.n_epochs):

            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            scheduler.step()

            # Test AUC.
            auc, (val_loss_pos, val_loss_neg) = self.test(dataset, ae_net, verbose=False)

            if auc > best_auc:
                best_auc = auc
                self.best_weights = {}
                for key in ae_net.state_dict():
                    self.best_weights[key] = ae_net.state_dict()[key].cpu()

            avg_loss = loss_epoch / n_batches

            # Do Tensorboard logging.
            self.writer.add_scalar('ae/loss', avg_loss, epoch)
            self.writer.add_scalar('ae/AUC', auc, epoch)
            self.writer.add_scalar('ae/loss_val_pos', val_loss_pos, epoch)
            self.writer.add_scalar('ae/loss_val_neg', val_loss_neg, epoch)

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}\t AUC: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, avg_loss, auc))

        pretrain_time = time.time() - start_time
        logger.info('Pretraining time: %.3f' % pretrain_time)
        logger.info('Finished pretraining.')

        return ae_net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet, verbose=True):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        if verbose:
            logger.info('Testing autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                n_batches += 1

        if verbose:
            logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))

        n_pos = 0; loss_pos = 0
        n_neg = 0; loss_neg = 0
        for idx, label, score in idx_label_score:
            if label:
                n_pos += 1
                loss_pos += score
            else:
                n_neg += 1
                loss_neg += score
        loss_pos /= n_pos
        loss_neg /= n_neg

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = -np.array(scores)


        auc = roc_auc_score(labels, scores)
        if verbose:
            roc_x, roc_y, roc_thres = roc_curve(labels, scores)
            thres_best = computeMaxYoudensIndex(roc_x, roc_y, roc_thres)
            self.thres = -thres_best
            logger.info('Test set AUC: {:.2f}%'.format(100. * auc))

            test_time = time.time() - start_time
            logger.info('Autoencoder testing time: %.3f' % test_time)
            logger.info('Finished testing autoencoder.')

        return auc, (loss_pos, loss_neg)
