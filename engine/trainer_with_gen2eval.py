from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import Trainer

class TrainerWithGenToEval(Trainer):
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys: list[str] = None,
    ): 
        with torch.no_grad(), self.compute_loss_context_manager():
            inputs = self._prepare_inputs(inputs)
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                return (loss, None, None)
            sample_idxs = inputs.pop('sample_idxs')
            evaluation_kwargs = inputs.pop('evaluation_kwargs')
            evaluator = evaluation_kwargs.pop('evaluator')
            # print(type(model))
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            output_ids = getattr(model, evaluator)(**inputs, **evaluation_kwargs, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            return (None, output_ids.reshape(1, -1), sample_idxs)
    
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            compute_metrics_dict = self.compute_metrics
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                self.compute_metrics = compute_metrics_dict[eval_dataset_name]
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            self.compute_metrics = compute_metrics_dict
            return metrics
        else:
            return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)