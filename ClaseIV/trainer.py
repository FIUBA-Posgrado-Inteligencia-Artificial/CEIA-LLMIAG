"""
trainer

Author: Abraham Rodriguez \n

CreationDate: 24/5/2023 
UpdateDate: 1/6/2025

This module provides utilities for training PyTorch models, including early stopping,
checkpointing, mixed precision support, and training/evaluation loops.

This module is heavily inspired on Huggingface's Trainer/Accelerate and Pytorch Lightning, for relevant docs read:
- https://docs.pytorch.org/docs/stable/amp.html
- https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
- https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
- https://pytorch-lightning.readthedocs.io/en/1.5.10/advanced/training_tricks.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- https://lightning.ai/docs/pytorch/1.6.5/common/checkpointing.html
"""
import copy
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import os

import warnings
import functools

def deprecated(reason):
	"""
    Decorator to mark functions as deprecated. It will emit a warning when the function is used.

    Args:
        reason (str): The reason why the function is deprecated.
	"""
	def decorator(func):
		@functools.wraps(func)
		def wrapped(*args, **kwargs):
			warnings.warn(
				f"{func.__name__}() is deprecated: {reason}",
				category=DeprecationWarning,
				stacklevel=2
			)
			return func(*args, **kwargs)
		return wrapped
	return decorator


class EarlyStopping():
	"""
	EarlyStopping serves as a mechanism to check if the loss does not have a considerable change, this can help to prevent overfitting
	and reduce the number of epochs (training time).
	"""
	def __init__(self, patience:int=5, min_delta :float=0, restore_best_weights:bool=True):
		"""

		Class constructor, sets mechanism to a certain quantity of patience, and a defined min_delta,
		and the best weights of the trained model.

		:param patience : patience to stop
		:type patience : int

		:param min_delta : minimum difference between losses per epoch.
		:type min_delta : float

		:param restore_best_weights :  restore best model
		:type restore_best_weights : bool

		"""
		self.patience = patience
		self.min_delta = min_delta
		self.restore_best_weights = restore_best_weights
		self.best_model = None
		self.best_loss = None
		self.counter = 0
		self.status = ""

	def __call__(self, model:torch.nn.Module, val_loss: float):
		"""
		Excutes logic when calling EarlyStopping object e.g
		es = EarlyStopping(patience=5)
		es(model,val_loss)
		"""
		if self.best_loss is None:
			self.best_loss = val_loss
			self.best_model = copy.deepcopy(model)

		elif self.best_loss - val_loss > self.min_delta:
			self.best_loss = val_loss
			self.counter = 0
			self.best_model.load_state_dict(model.state_dict())

		elif self.best_loss - val_loss < self.min_delta:
			self.counter += 1
			if self.counter >= self.patience:
				self.status = f"Stopped on {self.counter}"
				if self.restore_best_weights:
					model.load_state_dict(self.best_model.state_dict())
				return True

		self.status = f"{self.counter}/{self.patience}"
		return False

class Trainer():
	"""
	Custom trainer class that wraps model training and evaluation using PyTorch.

	This class supports:
	- Automatic Mixed Precision (AMP)
	- Gradient accumulation
	- Checkpointing during training
	"""
	def __init__(self,
			  	model : torch.nn.Module,
				train_data_loader: DataLoader,
				test_data_loader: DataLoader,
				loss_fn:torch.nn.Module,
				gradient_accumulation_steps :int,
				optimizer: torch.optim.Optimizer,
				scheduler: torch.optim.lr_scheduler.LRScheduler,
				device: str,
				save_dir : str = "./checkpoint",
				save_every_n = 1000
				):
		"""
        Initializes the Trainer class with the given model, data loaders, optimizer, and other training utilities.

        Args:
            model (torch.nn.Module): The model to train and evaluate.
            train_data_loader (DataLoader): DataLoader for the training dataset.
            test_data_loader (DataLoader): DataLoader for the test or validation dataset.
            loss_fn (torch.nn.Module): Loss function to optimize.
            gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating weights.
            optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
            scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler.
            device (str): Device to run the model on ('cpu' or 'cuda').
			save_dir (str): checkpoint directory, defaults to"./checkpoint",
			save_every_step (int): save every N steps, defaults to 1000

        """
		self.model = model
		self.train_data_loader = train_data_loader
		self.test_data_loader = test_data_loader
		self.loss_fn = loss_fn
		self.optimizer = optimizer
		self.device = device
		self.gradient_accumulation_steps = gradient_accumulation_steps
		self.scheduler = scheduler
		self.save_dir = save_dir
		self.save_every = save_every_n

	def save_checkpoint(self, step: int, final=False):
		"""
		Saves a training checkpoint to disk, including model, optimizer, scheduler,
		and (if available) scaler state dictionaries.

		Args:
			step (int): The current training step or epoch number to include in the checkpoint.
			final (bool, optional): If True, saves the checkpoint as the final model.
									The filename will use 'final' instead of the step number.
									Default is False.

		Notes:
			- Checkpoints are saved under `self.save_dir` with a filename format:
				- `checkpoint_step_{step}.pt` for intermediate steps.
				- `checkpoint_final.pt` for the final checkpoint.
			- Includes `scaler_state_dict` only if `self.scaler` exists (used in AMP training).
			- Creates the save directory if it does not already exist.
		"""
		os.makedirs(self.save_dir, exist_ok=True)
		suffix = f"final" if final else f"step_{step}"
		path = os.path.join(self.save_dir, f"checkpoint_{suffix}.pt")
		torch.save({
			"model_state_dict": self.model.state_dict(),
			"optimizer_state_dict": self.optimizer.state_dict(),
			"scaler_state_dict": self.scaler.state_dict() if hasattr(self, "scaler") else None,
			"scheduler_state_dict": self.scheduler.state_dict(),
			"step": step
		}, path)

	@deprecated("Use train_model_v2 for pretraining due to efficiency")
	def train_model(self,use_amp = False, dtype : torch.dtype = torch.bfloat16):

		model = self.model.train()
		scaler = torch.amp.GradScaler(enabled=use_amp)
		losses = []
		bar = tqdm(self.train_data_loader)
		for train_input, train_mask in bar:
				train_mask = train_mask.to(self.device)
				train_input=train_input.to(self.device)
				with torch.autocast(device_type=self.device, dtype=dtype, enabled=use_amp):
					output = model(train_input)
					loss = self.loss_fn(output, train_mask)
				if isinstance(dtype, type(torch.float16)):
					scaler.scale(loss).backward()
					scaler.step(self.optimizer)
					scaler.update()
				else:

					loss.backward()
					self.optimizer.step()

				# outputs=model(train_input.float())
				# loss = loss_fn(outputs.float(), train_mask.float())
				losses.append(loss.item())
				#loss.backward()
				#optimizer.step()
				#optimizer.zero_grad()
				for param in model.parameters():
					param.grad = None
				bar.set_description(f"loss {loss:.5f}")
		return np.mean(losses)


	def train_model_v2(self, use_amp: bool = False, dtype: torch.dtype = torch.bfloat16):
		"""
			Efficient training loop with optional Automatic Mixed Precision (AMP) support and gradient accumulation.

			This method performs one full pass over the training dataset using the given data loader.
			It supports Automatic Mixed Precision (AMP) training, gradient clipping, and gradient accumulation
			to handle larger effective batch sizes.

			Args:
				use_amp (bool, optional): If True, enables Automatic Mixed Precision (AMP) training.
										Default is False.
				dtype (torch.dtype, optional): The floating point precision to use when AMP is enabled.
											Common options include `torch.float16` or `torch.bfloat16`.
											Default is `torch.bfloat16`.

			Returns:
				float: The average of the last 10 training loss values.

			Notes:
				- Gradients are clipped to a maximum L2 norm of 1.0 to improve stability.
				- Gradients are accumulated across `self.gradient_accumulation_steps` batches before stepping.
				- At the end of training, a checkpoint is saved and CUDA memory is cleared.

			Raises:
				RuntimeError: If CUDA is not available while AMP is requested.
			"""
		model = self.model.train()
		scaler = torch.amp.GradScaler(device=self.device,enabled=use_amp)
		losses = []
		accumulation_count = 0
		global_step = 0
		bar = tqdm(self.train_data_loader)
		for train_input, train_mask in bar:

			train_mask = train_mask.to(self.device, non_blocking=True)
			train_input = train_input.to(self.device, non_blocking=True)
			with torch.autocast(device_type=self.device, dtype=dtype, enabled=use_amp):
				output = model(train_input)  # [B, T, vocab_size]
				B, T, C = output.shape
				loss = self.loss_fn(output.view(B * T, C), train_mask.view(B * T))
			loss_for_logging = loss.detach()
			loss = loss / self.gradient_accumulation_steps

			if use_amp and (dtype == torch.float16 or dtype == torch.bfloat16):
				scaler.scale(loss).backward()
			else:
				loss.backward()

			torch.nn.utils.clip_grad_norm_(model.parameters(), 1., norm_type=2)
			accumulation_count += 1
			global_step += 1

			if accumulation_count % self.gradient_accumulation_steps == 0:
				if use_amp and dtype == torch.float16:
					scaler.step(self.optimizer)
					scaler.update()
				else:
					self.optimizer.step()
				self.optimizer.zero_grad(set_to_none=True)
				self.scheduler.step()
				accumulation_count = 0

			losses.append(loss_for_logging)

			if len(losses) % 10 == 0:
				bar.set_description(f"loss {torch.mean(torch.stack(losses[-10:])).item():.5f}")

		# Final step if not aligned with accumulation
		if accumulation_count != 0:
			if use_amp and dtype == torch.float16:
				scaler.step(self.optimizer)
				scaler.update()
			else:
				self.optimizer.step()
			self.optimizer.zero_grad(set_to_none=True)
			self.scheduler.step()

		self.save_checkpoint(global_step, final=True)
		torch.cuda.empty_cache()
		return torch.mean(torch.stack(losses[-10:])).item()


	def eval_model(self):
		"""
		Evaluates the model on the test dataset.

		This method switches the model to evaluation mode, disables gradient computation,
		and computes the average loss over the test dataset.

		A progress bar is displayed using `tqdm` to show validation loss in real time.

		Returns:
			float: The mean loss over the entire test dataset.
		"""
		model = self.model.eval()

		losses = []
		bar = tqdm(self.test_data_loader)
		with torch.no_grad():
			for val_input, val_mask in bar:

				val_mask = val_mask.to(self.device)
				val_input = val_input.to(self.device)
				outputs = model(val_input)  # [B, T, vocab_size]
				B, T, C = outputs.shape
				loss = self.loss_fn(outputs.view(B * T, C), val_mask.view(B * T))

				losses.append(loss.item())
				bar.set_description(f"val_loss {loss:.5f}")

		return np.mean(losses)