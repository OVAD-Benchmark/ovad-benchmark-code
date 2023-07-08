import torch
from torch import nn
import torch.distributed as dist
from detectron2.utils.events import get_event_storage
from torch.nn import functional as F


def stats(tensor):
    t = tensor.cpu().detach().numpy()
    return {
        "device": tensor.device.index,
        "shape": tensor.shape,
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.to(torch.float32).mean()),
        "std": float(tensor.to(torch.float32).std()),
    }


class LoggedModule(nn.Module):
    def __init__(self):
        super(LoggedModule, self).__init__()
        self.log_info = {}
        self._log_print = False
        self._log_raise_nan = False

    def log(self, name, tensor):
        s = stats(tensor)
        self.log_info[name] = s
        if self._log_print:
            print(f"RANK {dist.get_rank()}: {name}", s)
        if self._log_raise_nan and torch.isnan(tensor).any():
            raise ValueError()

    def log_dict(self, d):
        self.log_info.update(d)
        if self._log_print:
            print(f"RANK {dist.get_rank()}: {d}")
        if self._log_raise_nan:
            for v in d.values():
                if torch.isnan(v).any():
                    raise ValueError()


def _log_attribute_stats(
    pred_logits, att_vectors, threshole=0.01, prefix="fast_rcnn_att"
):
    """
    Log the miltilabel classification metrics to EventStorage.

    Args:
        pred_logits: R x K logits.
        att_vectors: R x K labels, 0 - not present, 1 - present, -1 not known.
    """
    # print('pos_attributes', (att_vectors==1).sum())
    # print('neg_attributes', (att_vectors==0).sum())
    pos_attributes = (att_vectors == 1).sum() / att_vectors.shape[0]
    neg_attributes = (att_vectors == 0).sum() / att_vectors.shape[0]
    if pos_attributes > 0:
        storage = get_event_storage()
        storage.put_scalar(f"{prefix}/pos_attributes", pos_attributes)
        storage.put_scalar(f"{prefix}/neg_attributes", neg_attributes)
        storage.put_scalar(
            f"{prefix}/neg_pos_ratio", neg_attributes / (pos_attributes + 1e-7)
        )
