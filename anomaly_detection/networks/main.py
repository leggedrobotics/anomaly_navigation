from .stack_conv_net import StackConvNet, StackConvNet_Autoencoder
from .real_nvp import RealNVP, EncoderRealNVP


def build_network(net_name, cfg):
    """Builds the neural network."""

    implemented_networks = ('StackConvNet')
    assert net_name in implemented_networks

    net = None

    if net_name == 'StackConvNet':
        n_channel = 0
        if cfg.settings['rgb']:
            n_channel += 3
        if cfg.settings['ir']:
            n_channel += 1
        if cfg.settings['depth']:
            n_channel += 1
        if cfg.settings['depth_3d']:
            n_channel += 2
        if cfg.settings['normals']:
            n_channel += 2
        if cfg.settings['normal_angle']:
            n_channel += 1
        net = StackConvNet(in_channels=n_channel, 
                            use_bn=cfg.settings['batchnorm'], 
                            use_dropout=cfg.settings['dropout'])

    if cfg.settings['objective'] == 'real-nvp':
        net_nvp = RealNVP(in_dim=net.rep_dim, mid_dim=2*net.rep_dim)
        net = EncoderRealNVP(net, net_nvp)

    return net


def build_autoencoder(net_name, cfg):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('StackConvNet')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'StackConvNet':
        n_channel = 0
        if cfg.settings['rgb']:
            n_channel += 3
        if cfg.settings['ir']:
            n_channel += 1
        if cfg.settings['depth']:
            n_channel += 1
        if cfg.settings['depth_3d']:
            n_channel += 2
        if cfg.settings['normals']:
            n_channel += 2
        if cfg.settings['normal_angle']:
            n_channel += 1
        ae_net = StackConvNet_Autoencoder(in_channels=n_channel, 
                                           use_bn=cfg.settings['batchnorm'], 
                                           use_dropout=cfg.settings['dropout'])

    return ae_net
