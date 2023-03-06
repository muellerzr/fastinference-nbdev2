# Copyright 2023 Zachary Mueller. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import typing

import torch
from fastai.callback.core import Callback, GatherPredsCallback
from fastai.learner import Learner
from fastai.torch_core import apply, nested_reorder
from fastcore.foundation import L
from fastcore.xtras import ContextManagers
from torch.utils.data import DataLoader


class InferenceLearner(Learner):
    """
    A wrapper around a `Learner` that can be used for inference on a single item. Includes new methods for `predict`
    and `get_preds` that better handle inference, such as properly decoding the outputs completely during `get_preds`.

    The wrapped `Learner` object is available at `self.learn`
    """

    def __init__(self, learn: Learner):
        self.learn = learn
        self.dls = learn.dls

    def predict(self, item: typing.Any, with_input: bool = False) -> typing.List[torch.Tensor]:
        """
        Predict the output of a single item using the trained model. Will automatically set the number of workers to 0
        to avoid a slowdown in inference.

        Args:
            item (`any`):
                A single item to be predicted on that can be opened by the `type_transforms`
            with_input (`bool`, *optional*, defaults to False):
                If `True`, returns the input along with the prediction

        Returns:
            A list of up to three tensors: the input (if `with_input=True`), the prediction, and the decoded prediction
        """
        dataloader = self.dls.test_dl([item], num_workers=0)
        if with_input:
            input, probabilities, _, decoded_outputs = self.get_preds(
                dl=dataloader, with_input=with_input, with_decoded=True
            )
            return input, probabilities, decoded_outputs
        else:
            probabilities, _, decoded_outputs = self.get_preds(dl=dataloader, with_input=with_input, with_decoded=True)
            return probabilities, decoded_outputs

    def get_preds(
        self,
        dataloader_index: int = 1,
        dataloader: DataLoader = None,
        with_inputs: bool = False,
        with_decoded_predictions: bool = False,
        with_loss_values: bool = False,
        raw_values: bool = False,
        activation: typing.Callable = None,
        deshuffle_dataset: bool = True,
        callbacks: typing.List[Callback] = None,
        **kwargs,
    ):
        """
        Gets predictions on an input (either a dataloader existing in the `Learner` or a new `DataLoader` object) and
        potentially modifies their results.

        Args:
            dataloader_index (`int`, *optional*, defaults to 1):
                The index of the dataloader to use for inference stored in `self.dls`. If 0, will be the training
                dataloader. If 1, will be the validation dataloader.
            dataloader (`DataLoader`, *optional*, defaults to None):
                A new `DataLoader` object to use for inference. If passed in, will ignore `dataloader_index`.
            with_inputs (`bool`, *optional*, defaults to False):
                Whether to return the inputs along with the predictions
            with_decoded_predictions (`bool`, *optional*, defaults to False):
                Whether to also return the predictions passed through the loss function's `decode` method
            with_loss_values (`bool`, *optional*, defaults to False):
                Whether to also return the loss values
            raw_values (`bool`, *optional*, defaults to False):
                Whether to return the raw values or the probabilities
            activation (`typing.Callable`, *optional*, defaults to None):
                A function to apply to the predictions before returning them
            deshuffle_dataset (`bool`, *optional*, defaults to True):
                Whether to deshuffle the dataset before inference. Only applicable to fastai `DataLoader` objects and
                `Datasets`
            callbacks (`typing.List[Callback]`, *optional*, defaults to None):
                A list of `Callback` objects to use during inference
            kwargs:
                Additional keyword arguments to pass to the `GatherPredsCallback` callback
        """
        if dataloader is None:
            dataloader = self.dls[dataloader_index].new(shuffled=False, drop_last=False)
        if deshuffle_dataset and hasattr(dataloader, "get_idxs"):
            original_idxs = dataloader.get_idxs()
            dataloader = dataloader.new(get_idxs=lambda: original_idxs)
        preds_callback = GatherPredsCallback(with_input=with_inputs, with_loss=with_loss_values, **kwargs)
        context_managers = self.learn.validation_context(
            cbs=L(callbacks) + [preds_callback],
            inner=False,
        )
        if with_loss_values:
            context_managers.append(self.learn.loss_not_reduced())
        with ContextManagers(context_managers):
            self.learn._do_epoch_validate(dl=dataloader)
            if activation is not None:
                activation = getattr(self.learn.loss_func, "activation", lambda x: x)
            results = preds_callback.all_tensors()
            pred_index = 1 if with_inputs else 0
            if results[pred_index] is not None:
                if not raw_values:
                    results[pred_index] = activation(results[pred_index])
                if with_decoded_predictions:
                    decode_function = getattr(self.learn.loss_func, "decodes", lambda x: x)
                    results.insert(pred_index + 2, decode_function(results[pred_index]))
                if deshuffle_dataset and hasattr(dataloader, "get_idxs"):
                    results = nested_reorder(results, torch.tensor(original_idxs).argsort())
                if with_decoded_predictions:
                    if hasattr(self.dls, "categorize"):
                        results[pred_index + 1] = apply(self.dls.categorize.decode, [*results[pred_index + 2]])
                    elif hasattr(self.dls, "multi_categorize"):
                        results[pred_index + 1] = apply(self.dls.multi_categorize.decode, [*results[pred_index + 2]])
                return tuple(results)
            self.learn._end_cleanup()
