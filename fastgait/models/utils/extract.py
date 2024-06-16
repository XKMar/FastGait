import time
from collections import OrderedDict

import torch
import torch.nn.functional as F

from ...utils.dist_utils import all_gather_tensor, get_dist_info, synchronize
from ...utils.meters import Meters


@torch.no_grad()
def extract_features(
    model, 
    data_loader,  
    dataset,  
    cuda=True,  
    normalize=False,  
    print_freq=1000,  
    save_memory=False,
    for_testing=True,
    prefix="Extract: ",
):

    progress = Meters({"Time": ":.3f", "Data": ":.3f"}, len(data_loader), prefix=prefix)

    rank, world_size, is_dist = get_dist_info()
    features = []

    model.eval()
    data_iter = iter(data_loader)

    end = time.time()
    for i in range(len(data_loader)):
        data = next(data_iter)
        progress.update({"Data": time.time() - end})

        images = data["image"]
        if cuda:
            images = images.cuda()

        # compute output
        outputs = model(images)["feature"] # [B T H W]

        if isinstance(outputs, list) and for_testing:
            outputs = torch.cat(outputs, dim=-1)

        if normalize:
            if isinstance(outputs, list):
                outputs = [F.normalize(out, p=2, dim=-1) for out in outputs]
            outputs = F.normalize(outputs, p=2, dim=-1)

        if isinstance(outputs, list):
            outputs = torch.cat(outputs, dim=1).data.cpu()
        else:
            outputs = outputs.data.cpu()

        features.append(outputs)

        # measure elapsed time
        progress.update({"Time": time.time() - end})
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    synchronize()

    if is_dist and cuda:
        # distributed: gather features from all GPUs
        features = torch.cat(features)
        all_features = all_gather_tensor(features.cuda(), save_memory=save_memory)
        all_features = all_features.cpu()[: len(dataset)]

    else:
        # no distributed, no gather
        all_features = torch.cat(features, dim=0)[: len(dataset)]

    torch.cuda.empty_cache()
    
    return all_features