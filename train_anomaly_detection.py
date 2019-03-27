import click
import torch
import logging
import random
import numpy as np
import shutil
import os

from anomaly_detection.utils.config import Config
from anomaly_detection.utils.visualization.plot_images_grid import plot_images_grid
from anomaly_detection.anomalyDetection import AnomalyDetection
from anomaly_detection.datasets.main import load_dataset
from torch.utils.tensorboard import SummaryWriter


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['selfsupervised']))
@click.argument('net_name', type=click.Choice(['StackConvNet']))
@click.argument('xp_path', type=click.Path(exists=False))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--train_folder', '-t', type=str, default=None, multiple=True)
@click.option('--val_pos_folder', '-vp', type=str, default=None, multiple=True)
@click.option('--val_neg_folder', '-vn', type=str, default=None, multiple=True)
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--objective', type=click.Choice(['one-class', 'soft-boundary', 'real-nvp']), default='one-class',
              help='Specify Anomaly Detection objective ("one-class", "soft-boundary", or "real-nvp").')
@click.option('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for network training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for objective.')
@click.option('--pretrain', type=bool, default=True,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--rgb', is_flag=True)
@click.option('--ir', is_flag=True)
@click.option('--depth', is_flag=True)
@click.option('--depth_3d', is_flag=True)
@click.option('--normals', is_flag=True)
@click.option('--normal_angle', is_flag=True)
@click.option('--batchnorm', is_flag=True)
@click.option('--dropout', is_flag=True)
@click.option('--augment', is_flag=True)
@click.option('--fix_encoder', is_flag=True)
def main(dataset_name, net_name, xp_path, data_path, train_folder, val_pos_folder, val_neg_folder, load_config, 
         load_model, objective, nu, device, seed, optimizer_name, lr, n_epochs, lr_milestone, batch_size, 
         weight_decay, pretrain, ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, 
         ae_weight_decay, n_jobs_dataloader, normal_class, rgb, ir, depth, depth_3d, normals, normal_angle, 
         batchnorm, dropout, augment, fix_encoder):
    main_func(dataset_name, net_name, xp_path, data_path, train_folder, val_pos_folder, val_neg_folder, load_config, 
              load_model, objective, nu, device, seed, optimizer_name, lr, n_epochs, lr_milestone, batch_size, 
              weight_decay, pretrain, ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, 
              ae_weight_decay, n_jobs_dataloader, normal_class, rgb, ir, depth, depth_3d, normals, normal_angle, 
              batchnorm, dropout, augment, fix_encoder)



def main_func(dataset_name, net_name, xp_path, data_path, train_folder, val_pos_folder, val_neg_folder, load_config, 
              load_model, objective, nu, device, seed, optimizer_name, lr, n_epochs, lr_milestone, batch_size, 
              weight_decay, pretrain, ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, 
              ae_weight_decay, n_jobs_dataloader, normal_class, rgb, ir, depth, depth_3d, normals, normal_angle, 
              batchnorm, dropout, augment, fix_encoder):

    # Get configuration
    cfg = Config(locals().copy())

    assert rgb or ir or depth or depth_3d or normals or normal_angle, 'Need to select at least one input channel'

    # Get logging name based on settings.
    if net_name=='StackConvNet':
        log_folder = 'stack'
    if rgb:
        log_folder += '_rgb'
    if depth:
        log_folder += '_depth'
    if ir:
        log_folder += '_ir'
    if depth_3d:
        log_folder += '_3d'
    if normals:
        log_folder += '_normals'
    if normal_angle:
        log_folder += '_ang'
    if batchnorm:
        log_folder += '_bn'
    if dropout:
        log_folder += '_drop'
    if augment:
        log_folder += '_aug'
    if not pretrain:
        log_folder += '_nopre'
    if objective == 'one-class':
        log_folder += '_hard'
    elif objective == 'soft-boundary':
        log_folder += '_soft'
    elif objective == 'real-nvp':
        log_folder += '_nvp'
    if fix_encoder:
        log_folder += '_fix'
    if ae_n_epochs != 350 or n_epochs != 150:
        log_folder += '_' + str(ae_n_epochs) + '_' + str(n_epochs)

    tb_path = os.path.join(xp_path, 'tb', log_folder)
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    else:
        for file in os.listdir(tb_path):
            file_path = os.path.join(tb_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    xp_path = os.path.join(xp_path, log_folder)

    if not os.path.exists(xp_path):
        os.makedirs(xp_path)


    writer = SummaryWriter(tb_path)

    # Copy executed script to log folder.
    shutil.copyfile('train_anomaly_detection.sh', os.path.join(xp_path, 'train_anomaly_detection.sh'))

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print configuration
    logger.info('Anomaly detection objective: %s' % cfg.settings['objective'])
    logger.info('Nu-paramerter: %.2f' % cfg.settings['nu'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class, cfg)

    # Initialize model and set neural network \phi
    anomaly_detection = AnomalyDetection(writer, cfg.settings['objective'], cfg.settings['nu'])
    anomaly_detection.set_network(net_name, cfg)
    # If specified, load model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        anomaly_detection.load_model(model_path=load_model, cfg=cfg, load_ae=pretrain)
        logger.info('Loading model from %s.' % load_model)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # Pretrain model on dataset (via autoencoder)
        anomaly_detection.pretrain(dataset,
                           cfg,
                           optimizer_name=cfg.settings['ae_optimizer_name'],
                           lr=cfg.settings['ae_lr'],
                           n_epochs=cfg.settings['ae_n_epochs'],
                           lr_milestones=cfg.settings['ae_lr_milestone'],
                           batch_size=cfg.settings['ae_batch_size'],
                           weight_decay=cfg.settings['ae_weight_decay'],
                           device=device,
                           n_jobs_dataloader=n_jobs_dataloader)

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    anomaly_detection.train(dataset,
                    augment=cfg.settings['augment'],
                    optimizer_name=cfg.settings['optimizer_name'],
                    lr=cfg.settings['lr'],
                    n_epochs=cfg.settings['n_epochs'],
                    lr_milestones=cfg.settings['lr_milestone'],
                    batch_size=cfg.settings['batch_size'],
                    weight_decay=cfg.settings['weight_decay'],
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader,
                    fix_encoder=cfg.settings['fix_encoder'])

    # Test model
    anomaly_detection.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Plot most anomalous and most normal (within-class) test samples
    indices, labels, scores = zip(*anomaly_detection.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score

    if dataset_name in ('selfsupervised'):

        if dataset_name == 'selfsupervised':
            if rgb:
                X_normals = torch.tensor(dataset.test_set.images[idx_sorted[:32], 0:3, ...])
                X_outliers = torch.tensor(dataset.test_set.images[idx_sorted[-32:], 0:3, ...])
            else:
                X_normals = torch.tensor(dataset.test_set.images[idx_sorted[:32], 0:1, ...])
                X_outliers = torch.tensor(dataset.test_set.images[idx_sorted[-32:], 0:1, ...])

        plot_images_grid(X_normals, export_img=xp_path + '/normals', title='Most normal examples', padding=2)
        plot_images_grid(X_outliers, export_img=xp_path + '/outliers', title='Most anomalous examples', padding=2)
        anomaly_detection.trainer.roc_plt[0].savefig(os.path.join(xp_path, 'roc_curve.svg'))

    # Save results, model, and configuration
    anomaly_detection.save_results(export_json=xp_path + '/results.json')
    anomaly_detection.save_model(export_model=xp_path + '/model.tar', export_best_model=os.path.join(xp_path, 'model_best.tar'), save_ae=pretrain)
    cfg.save_config(export_json=xp_path + '/config.json')

    return anomaly_detection


if __name__ == '__main__':
    main()
