import math
import os

import torch
import torch.distributed as dist


__LOG10 = math.log(10)


def mse2psnr(x):
    return -10.*torch.log(x)/__LOG10


def init_ddp():
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError:
        return 0, 1  # Single GPU run
        
    dist.init_process_group(backend="nccl")
    print(f'Initialized process {local_rank} / {world_size}')
    torch.cuda.set_device(local_rank)

    setup_dist_print(local_rank == 0)
    return local_rank, world_size


def setup_dist_print(is_main):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def using_dist():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not using_dist():
        return 1
    return dist.get_world_size()


def get_rank():
    if not using_dist():
        return 0
    return dist.get_rank()


def gather_all(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return [tensor]

    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)

    return tensor_list


def reduce_dict(input_dict, average=True):
    """
    Reduces the values in input_dict across processes, when distributed computation is used.
    In all processes, the dicts should have the same keys mapping to tensors of identical shape.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    keys = sorted(input_dict.keys())
    values = [input_dict[k] for k in keys] 

    if average:
        op = dist.ReduceOp.AVG
    else:
        op = dist.ReduceOp.SUM

    for value in values:
        dist.all_reduce(value, op=op)

    reduced_dict = {k: v for k, v in zip(keys, values)}

    return reduced_dict


def compute_adjusted_rand_index(true_mask, pred_mask):
    """
    Computes the adjusted rand index (ARI) of a given image segmentation, ignoring the background.
    Implementation following https://github.com/deepmind/multi_object_datasets/blob/master/segmentation_metrics.py#L20
    Args:
        true_mask: Tensor of shape [batch_size, n_true_groups, n_points] containing true
            one-hot coded cluster assignments, with background being indicated by zero vectors.
        pred_mask: Tensor of shape [batch_size, n_pred_groups, n_points] containing predicted
            cluster assignments encoded as categorical probabilities.
    """
    batch_size, n_true_groups, n_points = true_mask.shape
    n_pred_groups = pred_mask.shape[1]

    if n_points <= n_true_groups and n_points <= n_pred_groups:
        raise ValueError(
          "adjusted_rand_index requires n_groups < n_points. We don't handle "
          "the special cases that can occur when you have one cluster "
          "per datapoint.")

    true_group_ids = true_mask.argmax(1)
    pred_group_ids = pred_mask.argmax(1)

    # Convert to one-hot ('oh') representations
    true_mask_oh = true_mask.float()
    pred_mask_oh = torch.eye(n_pred_groups).to(pred_mask)[pred_group_ids].transpose(1, 2)

    n_points_fg = true_mask_oh.sum((1, 2))

    nij = torch.einsum('bip,bjp->bji', pred_mask_oh, true_mask_oh)

    nij = nij.double()  # Cast to double, since the expected_rindex can introduce numerical inaccuracies

    a = nij.sum(1)
    b = nij.sum(2)

    rindex = (nij * (nij - 1)).sum((1, 2))
    aindex = (a * (a - 1)).sum(1)
    bindex = (b * (b - 1)).sum(1)
    expected_rindex = aindex * bindex / (n_points_fg * (n_points_fg - 1))
    max_rindex = (aindex + bindex) / 2

    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    # We can get NaN in case max_rindex == expected_rindex. This happens when both true and
    # predicted segmentations consist of only a single segment. Since we are allowing the
    # true segmentation to contain zeros (i.e. background) which we ignore, it suffices
    # if the foreground pixels belong to a single segment.

    # We check for this case, and instead set the ARI to 1.

    def _fg_all_equal(values, bg):
        """
        Check if all pixels in values that do not belong to the background (bg is False) have the same
        segmentation id.
        Args:
            values: Segmentations ids given as integer Tensor of shape [batch_size, n_points]
            bg: Binary tensor indicating background, shape [batch_size, n_points]
        """
        fg_ids = (values + 1) * (1 - bg.int()) # Move fg ids to [1, n], set bg ids to 0
        example_fg_id = fg_ids.max(1, keepdim=True)[0]  # Get the id of an arbitrary fg cluster.
        return torch.logical_or(fg_ids == example_fg_id[..., :1],  # All pixels should match that id...
                                bg  # ...or belong to the background.
                               ).all(-1)

    background = (true_mask.sum(1) == 0)
    both_single_cluster = torch.logical_and(_fg_all_equal(true_group_ids, background),
                                            _fg_all_equal(pred_group_ids, background))

    # Ensure that we are only (close to) getting NaNs in exactly the case described above.
    matching = (both_single_cluster == torch.isclose(max_rindex, expected_rindex))

    if not matching.all().item():
        offending_idx = matching.int().argmin()

    return torch.where(both_single_cluster, torch.ones_like(ari), ari)


