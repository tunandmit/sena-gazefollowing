import os
import sys
import logging
import torch
from torch import nn
from typing import Dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    AutoImageProcessor,
    HfArgumentParser
)
from transformers.models.conditional_detr.modeling_conditional_detr import ConditionalDetrHungarianMatcher

from .argument import (
    ModelArguments,
    DataArguments,
)
from .data.dataloader import DataProcessor, SenDataProcessor, DataCollator
from .modeling import SenConDeTr
from .metrics import GazePointDistance, GazeHeatmapAUC


logger = logging.getLogger(__name__)



class SensiEvaluate(nn.Module):
    def __init__(self, matcher, evals:Dict):
        super(SensiEvaluate, self).__init__()
        
        self.matcher = matcher
        self.evals = evals
        
    # Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss._get_source_permutation_idx
    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        
        return batch_idx, source_idx
        
    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        
        idx = self._get_source_permutation_idx(indices)
        
        for _, fn in self.evals.items():
            fn(
                outputs,
                targets,
                indices,
                src_permutation_idx=idx,
            )
            
    def reset(self):
        for _, fn in self.evals.items():
            fn.reset_metrics()

    def get_metrics(self):
        metrics = {}
        for eval_name, fn in self.evals.items():
            metrics.update(fn.get_metrics())

        return metrics
    
    
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()
        
    #######################################################################
    #                                                                     #
    #######################################################################
    
    # Prepare processor
    image_processor = AutoImageProcessor.from_pretrained(model_args.model_name_or_path)
    
    # prepare data
    dataset_training, id2label, label2id  = SenDataProcessor(image_processor, data_args).__call__()
    
    sensi_loader = DataLoader(
        dataset_training['test'],
        batch_size=12,
        collate_fn=DataCollator(image_processor),
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )
    
    #######################################################################
    #                             Model                                   #
    #######################################################################
    
    model = SenConDeTr.from_pretrained(
        model_args.model_name_or_path,
        num_labels = len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        model_args=model_args
    )
    
    matcher = ConditionalDetrHungarianMatcher(
        class_cost=model.config.class_cost, 
        bbox_cost=model.config.bbox_cost, 
        giou_cost=model.config.giou_cost
    )
    
    eval_sena = SensiEvaluate(
        matcher = matcher,
        evals = {
            'GazePoint' : GazePointDistance(),
            'GazeHeapmapAUC': GazeHeatmapAUC()
            }   
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    all_result = {}
    fp16 = torch.cuda.is_available()
    result = []
    for batch in tqdm(sensi_loader):
        with torch.cuda.amp.autocast() if fp16 else nullcontext():
            with torch.no_grad():
                # set device
                for k, v in batch.items():
                    if isinstance(v, (tuple, list)):
                        for x in v:
                            for ki, vi in x.items():
                                x[ki] = vi.to(model.device)
                    else:
                        batch[k] = v.to(model.device)
                
                # predict
                model_output = model(
                    pixel_values = batch['pixel_values'],
                    pixel_mask = batch['pixel_mask'],
                )
                
                eval_sena(model_output, batch['labels'])

    print(eval_sena.get_metrics())

if __name__ == '__main__':
    main()