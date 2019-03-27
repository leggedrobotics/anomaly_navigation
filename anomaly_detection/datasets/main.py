from .selfsupervised_patches import SelfSupervisedDataset


def load_dataset(dataset_name, data_path, normal_class, cfg):
    """Loads the dataset."""

    implemented_datasets = ('selfsupervised')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'selfsupervised':
        dataset = SelfSupervisedDataset(root=data_path, 
                                        train=cfg.settings['train_folder'],
                                        val_pos=cfg.settings['val_pos_folder'],
                                        val_neg=cfg.settings['val_neg_folder'],
                                        rgb=cfg.settings['rgb'], 
                                        ir=cfg.settings['ir'], 
                                        depth=cfg.settings['depth'],
                                        depth_3d=cfg.settings['depth_3d'],
                                        normals=cfg.settings['normals'], 
                                        normal_angle=cfg.settings['normal_angle'])

    return dataset
