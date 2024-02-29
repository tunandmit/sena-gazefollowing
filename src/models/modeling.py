import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Optional, Union, Tuple, Dict
from dataclasses import dataclass

from transformers.models.conditional_detr.modeling_conditional_detr import (
    ConditionalDetrObjectDetectionOutput,
    ConditionalDetrHungarianMatcher,
    ConditionalDetrPreTrainedModel,
    ConditionalDetrModel,
    ConditionalDetrLoss
)
from transformers.models.conditional_detr.configuration_conditional_detr import (
    ConditionalDetrConfig
)

from .MLP import MLP
from ..argument import ModelArguments


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@dataclass
class GazeTransformerOutputWithLogs(ConditionalDetrObjectDetectionOutput):
    logs: Optional[Dict[str, any]] = None
    pred_gaze_boxes: Optional[Dict[str, any]] = None
    pred_gaze_logits: Optional[Dict[str, any]] = None
    pred_gaze_vectors: Optional[Dict[str, any]] = None
    pred_gaze_cone: Optional[Dict[str, any]] = None
    pred_gaze_heatmap: Optional[Dict[str, any]] = None
    pred_gaze_watch_outside: Optional[Dict[str, any]] = None
    objects_scores: Optional[Dict[str, any]] = None


class SenAGaze(ConditionalDetrPreTrainedModel):
    def __init__(
        self,
        config: ConditionalDetrConfig,
        model_args: ModelArguments,
    ):
        super().__init__(config)

        self.model_args = model_args

        # CONDITIONAL DETR encoder-decoder model
        self.model = ConditionalDetrModel(config)

        # Object detection heads
        self.class_labels_classifier = nn.Linear(
            config.d_model, config.num_labels
        )  # We add one for the "no object" class

        self.bbox_predictor = MLP(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            output_dim=4,
            num_layers=3
        )

        # Setup heatmap and watch outside MLPs
        self.gaze_heatmap_embed = MLP(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            output_dim=self.model_args.gaze_heatmap_size**2,
            num_layers=5,
        )

        self.gaze_watch_outside= MLP(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            output_dim=1,
            num_layers=1
        )

        # Initialize weights and apply final processing
        self.post_init()

    # taken from https://github.com/Atten4Vis/conditionalDETR/blob/master/models/conditional_detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        labels: Optional[List[dict]] = None,
        return_dict: Optional[bool] = None,
        **kwarg,
    ) -> Union[Tuple[torch.FloatTensor], ConditionalDetrObjectDetectionOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # First, sent images through CONDITIONAL_DETR base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            **kwarg
        )
        
        outputs['last_hidden_state'] = outputs['last_hidden_state'][:, : self.model_args.num_gaze_queries]
        outputs['reference_points'] = outputs['reference_points'][: self.model_args.num_gaze_queries]

        logits, pred_boxes, _ = self.logits_wrapper(outputs['last_hidden_state'], outputs['reference_points'])

        outputs_gaze_watch_outside = self.gaze_watch_outside(outputs['last_hidden_state']).sigmoid()
        outputs_gaze_heatmap = self.gaze_heatmap_embed(outputs['last_hidden_state']).sigmoid()

        gaze_outputs = {
            'logits' : logits,
            'pred_boxes' : pred_boxes,
            'outputs_gaze_watch_outside' : outputs_gaze_watch_outside,
            'outputs_gaze_heatmap' : outputs_gaze_heatmap,
        }

        if labels is not None:
            loss, loss_dict, logs = self.get_losses(gaze_outputs, labels)
        else:
            loss, loss_dict, logs = None, None, None

        # get loss
        if not return_dict:
            output = (logits, pred_boxes) + outputs
            return ((loss, loss_dict) + output) if loss is not None else output

        result= GazeTransformerOutputWithLogs(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            pred_gaze_heatmap=outputs_gaze_heatmap,
            pred_gaze_watch_outside=outputs_gaze_watch_outside,
            logs=logs
        )
        return result

    def logits_wrapper(self, last_hidden_state, reference_points):
        # class logits + predicted bounding boxes
        logits = self.class_labels_classifier(last_hidden_state)

        # pred bbox
        reference_before_sigmoid = inverse_sigmoid(reference_points).transpose(0, 1)
        tmp = self.bbox_predictor(last_hidden_state)
        tmp[..., :2] += reference_before_sigmoid
        pred_boxes = tmp.sigmoid()

        return logits, pred_boxes, reference_before_sigmoid

    def get_losses(self, outputs, labels):
        loss, loss_dict = None, None
        if labels is not None:
            # First: create the matcher
            matcher = ConditionalDetrHungarianMatcher(
                class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost
            )
            # Second: create the criterion
            losses = ["labels", "boxes", "cardinality", "heatmaps", "watch_outside"]
            criterion = SensiConditionalDetrLoss(
                matcher=matcher,
                num_classes=self.config.num_labels,
                focal_alpha=self.config.focal_alpha,
                losses=losses,
            ).to(self.device)

            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = outputs['logits']
            outputs_loss["pred_boxes"] = outputs['pred_boxes']
            outputs_loss["outputs_gaze_watch_outside"] = outputs['outputs_gaze_watch_outside']
            outputs_loss["outputs_gaze_heatmap"] = outputs['outputs_gaze_heatmap']

            loss_dict = criterion(outputs_loss, labels)

            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {"loss_ce": self.config.cls_loss_coefficient, "loss_bbox": self.config.bbox_loss_coefficient}
            weight_dict["loss_giou"] = self.config.giou_loss_coefficient
            weight_dict["loss_heatmaps"] = 2
            weight_dict["loss_watch_outside"] = 1

            results = []
            logs = dict()
            for k, _ in loss_dict.items():
              if k in weight_dict:
                result = loss_dict[k] * weight_dict[k]
                results.append(result)
                logs[k] = result.item()

            loss = sum(results)

        return loss, loss_dict, logs


class SensiConditionalDetrLoss(ConditionalDetrLoss):
    def __init__(self, matcher, num_classes, focal_alpha, losses):
        super(SensiConditionalDetrLoss, self).__init__(matcher, num_classes, focal_alpha, losses)
        self.gaze_loss_fn =  nn.MSELoss()
        self.heatmap_loss_fn = nn.MSELoss(reduction="none")

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "heatmaps": self.loss_heatmaps,
            "watch_outside" : self.loss_watch_outside
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")

        result = loss_map[loss](outputs, targets, indices, num_boxes)

        return result

    def loss_gaze_vector(self, outputs, targets, indices, num_boxes):
        if "pred_gaze_vectors" not in outputs:
            raise KeyError("No predicted gaze vector found in outputs")

        idx = self._get_source_permutation_idx(indices)
        tgt_padding = torch.cat(
            [t['regression_padding'][i] for t, (_, i) in zip (targets, indices)], dim=0
        ).squeeze(1)

        tgt_gaze_vectors = torch.cat(
            [t['gaze_vector'][i] for t, (_, i) in zip (targets, indices)], dim=0
        ).mean(dim=1)[~tgt_padding]

        src_gaze_vectors = outputs['pred_gaze_vectors'][idx][~tgt_padding]
        loss = self.gaze_loss_fn(src_gaze_vectors, tgt_gaze_vectors)

        return {"loss_gaze_vector" : loss}

    def loss_heatmaps(self, outputs, targets, indices, num_boxes):
        if "outputs_gaze_heatmap" not in outputs:
            raise KeyError("No predicted gaze heatmap found in outputs")

        idx = self._get_source_permutation_idx(indices)
        tgt_padding = torch.cat(
            [t['regression_padding'][i] for t, (_, i) in zip (targets, indices)], dim=0
        ).squeeze(1)
        tgt_watch_inout = torch.cat(
            [t['gaze_watch_outside'][i] for t, (_, i) in zip (targets, indices)], dim=0
        )[~tgt_padding]
        tgt_heatmap = torch.cat(
            [t['gaze_heatmaps'][i] for t, (_, i) in zip (targets, indices)], dim=0
        )[~tgt_padding].flatten(1, 2)
        tgt_inside = (tgt_watch_inout.argmax(-1) == 0).float()

        # If pred_gaze_heatmap is list, get the last one
        if isinstance(outputs["outputs_gaze_heatmap"], list):
            pred_heatmap = torch.stack(
                [outputs["outputs_gaze_heatmap"][i][j] for (i, j) in zip(idx[0], idx[1])]
            )
        else:
            pred_heatmap = outputs["outputs_gaze_heatmap"][idx]

        pred_heatmap = pred_heatmap[~tgt_padding]
        heatmap_loss = self.heatmap_loss_fn(pred_heatmap, tgt_heatmap).mean(dim=1)

        heatmap_loss = torch.mul(
            heatmap_loss, tgt_inside
        )  # Zero out loss when it's out-of-frame gaze case

        if tgt_inside.sum() > 0:
            heatmap_loss = heatmap_loss.sum() / tgt_inside.sum()
        else:
            heatmap_loss = heatmap_loss.sum()

        return {'loss_heatmaps' : heatmap_loss}

    def loss_watch_outside(self, outputs, targets, indices, num_boxes):
        if "outputs_gaze_watch_outside" not in outputs:
            raise KeyError("No predicted gaze watch outside found in outputs")

        idx = self._get_source_permutation_idx(indices)
        tgt_padding = torch.cat(
            [t['regression_padding'][i] for t, (_, i) in zip (targets, indices)], dim=0
        ).squeeze(1)
        tgt_watch_outside = torch.cat(
            [t['gaze_watch_outside'][i] for t, (_, i) in zip (targets, indices)], dim=0
        )[~tgt_padding].flatten()
        pred_watch_outside = outputs['outputs_gaze_watch_outside'][idx][~tgt_padding].flatten()
        loss = F.binary_cross_entropy_with_logits(pred_watch_outside, tgt_watch_outside.float())

        return {"loss_watch_outside" : loss}