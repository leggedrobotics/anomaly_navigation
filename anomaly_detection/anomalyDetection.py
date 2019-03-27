import json
import torch

from anomaly_detection.base.base_dataset import BaseADDataset
from anomaly_detection.networks.main import build_network, build_autoencoder
from anomaly_detection.optim.anomalyDetection_trainer import AnomalyDetectionTrainer
from anomaly_detection.optim.ae_trainer import AETrainer


class AnomalyDetection(object):

    def __init__(self, writer, objective: str = 'one-class', nu: float = 0.1):
        """Inits anomaly detection with one of the two objectives and hyperparameter nu."""

        assert objective in ('one-class', 'soft-boundary', 'real-nvp'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu
        self.R = 0.0  # hypersphere radius R
        self.Thres = None
        self.c = None  # hypersphere center c

        self.net_name = None
        self.net = None  # neural network \phi

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.auc_result = None

        self.writer = writer

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_auc_ae': None,
            'test_time': None,
            'test_scores': None,
            'thres_ae': 0,
        }

    def set_network(self, net_name, cfg):
        """Builds the neural network \phi."""
        self.net_name = net_name
        self.net = build_network(net_name, cfg)

    def train(self, dataset: BaseADDataset, augment: bool, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0, fix_encoder: bool = False):
        """Trains the anomaly detection model on the training data."""

        self.optimizer_name = optimizer_name
        self.trainer = AnomalyDetectionTrainer(self.writer, self.objective, self.R, self.c, self.nu, optimizer_name, lr=lr,
                                       n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                       weight_decay=weight_decay, device=device, n_jobs_dataloader=n_jobs_dataloader, 
                                       fix_encoder=fix_encoder)
        # Get the model
        self.net = self.trainer.train(dataset, self.net, augment)
        self.R = float(self.trainer.R.cpu().data.numpy())  # get float
        self.best_R = float(self.trainer.best_R.cpu().data.numpy())
        self.best_weights = self.trainer.best_weights
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get list
        self.results['train_time'] = self.trainer.train_time

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the anomaly detection model on the test data."""

        if self.trainer is None:
            self.trainer = AnomalyDetectionTrainer(self.objective, self.R, self.c, self.nu,
                                           device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(dataset, self.net)
        # Get results
        self.results['thres'] = self.trainer.thres
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_fpr5'] = self.trainer.test_fpr5
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def testInputSensitivity(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        if self.trainer is None:
            self.trainer = AnomalyDetectionTrainer(self.objective, self.R, self.c, self.nu,
                                           device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.testInputSensitivity(dataset, self.net)


    def pretrain(self, dataset: BaseADDataset, cfg, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the anomaly detection network \phi via autoencoder."""

        self.ae_net = build_autoencoder(self.net_name, cfg)
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(self.writer, optimizer_name, lr=lr, n_epochs=n_epochs, 
                                    lr_milestones=lr_milestones, batch_size=batch_size, 
                                    weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)
        self.results['test_auc_ae'], _ = self.ae_trainer.test(dataset, self.ae_net)
        self.results['thres_ae'] = self.ae_trainer.thres
        self.best_ae_weights = self.ae_trainer.best_weights
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        """Initialize the anomaly detection network weights from the encoder weights of the pretraining autoencoder."""

        if self.objective == 'real-nvp':
            net_dict = self.net.encoder.state_dict()
        else:
            net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict

        if self.objective == 'real-nvp':
            self.net.encoder.load_state_dict(net_dict)
        else:
            self.net.load_state_dict(net_dict)

    def save_model(self, export_model, export_best_model, save_ae=True):
        """Save anomaly detection model to export_model."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None
        ae_best_net_dict = self.best_ae_weights if save_ae else None
        print('Best threshold: ' + str(self.results['thres']))
        print('Best AE threshold: ' + str(self.results['thres_ae']))

        torch.save({'R': self.R,
                    'thres': self.results['thres'],
                    'thres_ae': self.results['thres_ae'],
                    'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)
        torch.save({'R': self.best_R,
                    'c': self.c,
                    'net_dict': self.best_weights,
                    'ae_net_dict': ae_best_net_dict}, export_best_model)

    def load_model(self, model_path, cfg, load_ae=False):
        """Load anomaly detection model from model_path."""

        model_dict = torch.load(model_path)

        self.R = model_dict['R']
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name, cfg)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
