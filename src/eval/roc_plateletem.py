import torch
import src.util as util
import random
import src.eval.config as config
import torchreg


def group_bounds_by_ground_truth(bound1, gt):
    topology_change_bounds = bound1[gt == 1].flatten().cpu().tolist()
    no_topology_change_bounds = bound1[gt == 0].flatten().cpu().tolist()

    # sample 10000 pixels for plotting (proportionally)
    K = len(topology_change_bounds) + len(no_topology_change_bounds)
    topology_change_bounds = random.sample(
        topology_change_bounds, k=int(len(topology_change_bounds) / K * 10000))
    no_topology_change_bounds = random.sample(
        no_topology_change_bounds, k=int(len(no_topology_change_bounds) / K * 10000))

    return topology_change_bounds, no_topology_change_bounds


def get_bounds_for_model_plateletem_dataset(model_name, bootstrap=False):
    # load model
    torchreg.settings.set_ndims(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = util.load_model_from_logdir(
        config.MODELS[model_name]["path"]["platelet-em"], model_cls=config.MODELS[model_name]["model_cls"])
    model.to(device)

    # load dataset
    datamodel = util.load_datamodule_for_model(model, )
    dataloader = datamodel.test_dataloader(bootstrap=bootstrap)

    # predict
    topology_change_bounds = []
    no_topology_change_bounds = []
    for batch in dataloader:
        I = batch['I0']['data'].to(device)
        J = batch['I1']['data'].to(device)
        gt = batch['Tcombined']['data'].to(device)
        _, bound_1, _, _ = model.bound(
            I, J, bidir=True)
        topology_change_bound, no_topology_change_bound = group_bounds_by_ground_truth(
            bound_1, gt)
        topology_change_bounds += topology_change_bound
        no_topology_change_bounds += no_topology_change_bound

    return topology_change_bounds, no_topology_change_bounds
