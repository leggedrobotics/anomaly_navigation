from anomaly_detection.base.base_trainer import BaseTrainer
from anomaly_detection.base.base_dataset import BaseADDataset
from anomaly_detection.base.base_net import BaseNet
from anomaly_detection.utils.eval_functions import computeMaxYoudensIndex, computeTprAt5Fpr
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'
import matplotlib.pyplot as plt
import cv2

import logging
import time
import torch
import torch.optim as optim
import numpy as np



# Adversarial training params.
EPS_AUGMENT = 2.5e-2




class AnomalyDetectionTrainer(BaseTrainer):

    def __init__(self, writer, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, fix_encoder: bool = False):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader, writer)

        assert objective in ('one-class', 'soft-boundary', 'real-nvp'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        self.fix_encoder = fix_encoder

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu
        self.thres = None

        # Do Real NVP initializations.
        

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet, augment: bool):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader, shuffle_train=False, shuffle_test=False)

        # Set optimizer (Adam optimizer for now)
        if self.fix_encoder:
            params = [param for name, param in net.named_parameters() if 'nvp' in name]
            param_names = [name for name, param in net.named_parameters() if 'nvp' in name]
            print('Optimizing params: ' + str(param_names))
        else:
            params = net.parameters()
        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Set prior with current center.
        self.prior = torch.distributions.MultivariateNormal(self.c, 
                                                 torch.eye(net.rep_dim, device=self.device))

        # Training
        best_auc = 0
        logger.info('Starting training...')
        start_time = time.time()
        for epoch in range(self.n_epochs):
            net.train()

            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                if augment:
                    # Adversarial augmentation.
                    inputs.requires_grad = True
                    outputs = net(inputs)
                    if self.objective == 'real-nvp':
                        z, log_det_J = outputs
                        log_prob_z = self.prior.log_prob(z)
                        loss = -log_prob_z.mean() - log_det_J.mean()
                    else:
                        dist = torch.sum((outputs - self.c) ** 2, dim=1)
                        if self.objective == 'soft-boundary':
                            scores = dist - self.R ** 2
                            loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                        elif self.objective == 'one-class':
                            loss = torch.mean(dist)
                    loss.backward()

                    r_adv = EPS_AUGMENT * inputs.grad.sign()
                    inputs.requires_grad = False
                    inputs_aug = inputs + r_adv
                else:
                    inputs_aug = inputs

                # cv2.imshow('rgb',inputs_aug[0,0:3,...].squeeze().cpu().numpy().transpose((1,2,0)))
                # cv2.imshow('ir',inputs_aug[3,0,...].squeeze().cpu().numpy())
                # cv2.imshow('depth',(inputs_aug[4,0,...]).squeeze().cpu().numpy())
                cv2.waitKey(0)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs_aug)
                if self.objective == 'real-nvp':
                    z, log_det_J = outputs
                    log_prob_z = self.prior.log_prob(z)
                    loss = -log_prob_z.mean() - log_det_J.mean()
                else:
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    if self.objective == 'soft-boundary':
                        scores = dist - self.R ** 2
                        loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                    else:   # 'one-class'
                        loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

            scheduler.step()

            # Test on val set.
            self.test(dataset, net, verbose=False)
            if self.test_auc > best_auc:
                best_auc = self.test_auc
                self.best_weights = {}
                for key in net.state_dict():
                    self.best_weights[key] = net.state_dict()[key].cpu()
                self.best_R = self.R.detach().clone()

            avg_loss = loss_epoch / n_batches

            # Do Tensorboard logging.
            self.writer.add_scalar('anomaly/loss', avg_loss, epoch)
            self.writer.add_scalar('anomaly/auc', self.test_auc, epoch)
            self.writer.add_scalar('anomaly/loss_val_pos', self.val_loss[0], epoch)
            self.writer.add_scalar('anomaly/loss_val_neg', self.val_loss[1], epoch)

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}\t AUC: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, avg_loss, self.test_auc))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet, verbose=True):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        if verbose:
            logger.info('Starting testing...')
            start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                if self.objective == 'real-nvp':
                    z, log_det_J = outputs
                    log_prob_z = self.prior.log_prob(z.squeeze())
                    scores = -log_prob_z - log_det_J.squeeze()    # We negate here, because we negate again later.
                else:
                    dist = torch.sum((outputs.squeeze() - self.c) ** 2, dim=1)
                    if self.objective == 'soft-boundary':
                        scores = dist - self.R ** 2
                    elif self.objective == 'one-class':
                        scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        # Compute pos and neg loss.
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


        if verbose:
            self.test_time = time.time() - start_time
            logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score
        self.val_loss = (loss_pos, loss_neg)

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        # Invert scores because we compute error, not class probability.
        scores = -scores

        self.test_auc = roc_auc_score(labels, scores)
        if verbose:
            logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

            # Plot ROC curve.
            roc_x, roc_y, roc_thres = roc_curve(labels, scores)
            thres_best = computeMaxYoudensIndex(roc_x, roc_y, roc_thres)
            self.test_fpr5 = computeTprAt5Fpr(roc_x, roc_y)
            if self.objective == 'real-nvp':
                thres_best = -thres_best
            self.thres = thres_best
            fig, ax = plt.subplots()
            ax.set(xlabel='FPR', ylabel='TPR', title='ROC curve: {:.2f}% AUC'.format(100.*self.test_auc))
            ax.plot(roc_x, roc_y)
            ax.grid()
            self.roc_plt = (fig, ax)
            # plt.show()
            plt.close()

            logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0).squeeze()

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def testInputSensitivity(self, dataset: BaseADDataset, net: BaseNet):
        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        net.eval()
        prop = GuidedBackProp(net)

        for inputs, labels, idx in test_loader:
            inputs = inputs.to(self.device)
            # Enable gradient wrt input.
            inputs.requires_grad = True
            outputs = net(inputs).squeeze()
            # Do backward pass.
            outputs.mean().backward(gradient=torch.ones(outputs.mean().size(), device=self.device))
            # Average over all but channels.
            sensitivity = inputs.grad
            sensitivity = sensitivity.abs().mean((0, 2, 3))
            print(sensitivity/sensitivity.norm())





def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
