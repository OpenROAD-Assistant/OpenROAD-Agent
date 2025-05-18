#BSD 3-Clause License
#
#Copyright (c) 2025, OpenROAD-Assistant
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import gc
from transformers import (
  Trainer,
  TrainerCallback
)

class ShuffleDatasetCallback(TrainerCallback):
  def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
    if train_dataloader is not None:
      seed = int(state.epoch) if state.epoch is not None else 118 #hard code the seed for reproducibility
      train_dataloader.dataset = train_dataloader.dataset.shuffle(seed=seed)
    return control

class StreamingDatasetCallback(TrainerCallback):
  def on_step_end(self, args, state, control, **kwargs):
    # Force garbage collection after every step to clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'memory_stats'):
      torch.cuda.reset_peak_memory_stats()
    return control
  
class CPUOffloadCallback(TrainerCallback):
  def on_step_end(self, args, state, control, model=None, **kwargs):
    if model is not None:
      for param in model.parameters():
        if param.grad is not None:
          param.grad.data = param.grad.data.cpu()
      
    gc.collect()
    torch.cuda.empty_cache()
    return control

class BatchedDataLoader:
  def __init__(self, dataset, batch_size=4):
    self.dataset = dataset
    self.batch_size = batch_size
    self.length = len(dataset)
    self.num_batches = (self.length + self.batch_size - 1) // self.batch_size
    
  def __iter__(self):
    self.current_batch = 0
    # Shuffle indices at the start of iteration
    self.indices = torch.randperm(self.length).tolist()
    return self
    
  def __next__(self):
    if self.current_batch >= self.num_batches:
      raise StopIteration
      
    # Get indices for the current batch
    start_idx = self.current_batch * self.batch_size
    end_idx = min(start_idx + self.batch_size, self.length)
    batch_indices = self.indices[start_idx:end_idx]
    self.current_batch += 1
    
    batch = [self.dataset[idx] for idx in batch_indices]
    batch = {"input_ids": torch.stack([item["input_ids"] for item in batch]),
             "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
             "labels": torch.stack([item["labels"] for item in batch])}
    gc.collect()
    torch.cuda.empty_cache()
    return batch
    
  def __len__(self):
    return self.num_batches

class CustomTrainer(Trainer):

  def get_train_dataloader(self):
    if self.train_dataset is None:
      raise ValueError("Trainer: training requires a train_dataset.")
      
    if hasattr(self.train_dataset, "set_format"):
      self.train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'], device='cpu')
    
    return BatchedDataLoader(
      dataset=self.train_dataset,
      batch_size=self.args.per_device_train_batch_size,
    )

  
  def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    labels = inputs.pop("labels")[..., 1:].contiguous()

    outputs = model(**inputs)
    logits = outputs["logits"][..., :-1, :].contiguous()

    loss = torch.nn.functional.cross_entropy(
      logits.view(-1, logits.shape[-1]),  
      labels.view(-1),                    
      reduction="mean",
      ignore_index=self.processing_class.pad_token_id
    )

    return (loss, outputs) if return_outputs else loss
  
