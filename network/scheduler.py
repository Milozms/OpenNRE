from functools import partial
import logging

class ReduceLROnPlateau(object):
	"""Reduce learning rate when a metric has stopped improving.
	Models often benefit from reducing the learning rate by a factor
	of 2-10 once learning stagnates. This scheduler reads a metrics
	quantity and if no improvement is seen for a 'patience' number
	of epochs, the learning rate is reduced.

	Args:
		optimizer (Optimizer): Wrapped optimizer.
		mode (str): One of `min`, `max`. In `min` mode, lr will
			be reduced when the quantity monitored has stopped
			decreasing; in `max` mode it will be reduced when the
			quantity monitored has stopped increasing. Default: 'min'.
		factor (float): Factor by which the learning rate will be
			reduced. new_lr = lr * factor. Default: 0.1.
		patience (int): Number of epochs with no improvement after
			which learning rate will be reduced. Default: 10.
		verbose (bool): If ``True``, prints a message to stdout for
			each update. Default: ``False``.
		threshold (float): Threshold for measuring the new optimum,
			to only focus on significant changes. Default: 1e-4.
		threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
			dynamic_threshold = best * ( 1 + threshold ) in 'max'
			mode or best * ( 1 - threshold ) in `min` mode.
			In `abs` mode, dynamic_threshold = best + threshold in
			`max` mode or best - threshold in `min` mode. Default: 'rel'.
		cooldown (int): Number of epochs to wait before resuming
			normal operation after lr has been reduced. Default: 0.
		min_lr (float or list): A scalar or a list of scalars. A
			lower bound on the learning rate of all param groups
			or each group respectively. Default: 0.
		eps (float): Minimal decay applied to lr. If the difference
			between new and old lr is smaller than eps, the update is
			ignored. Default: 1e-8.

	"""

	def __init__(self, mode='min', factor=0.1, patience=10,
				 verbose=False, threshold=1e-4, threshold_mode='rel',
				 cooldown=0, min_lr=0, eps=1e-8):

		if factor >= 1.0:
			raise ValueError('Factor should be < 1.0.')
		self.factor = factor

		self.patience = patience
		self.verbose = verbose
		self.cooldown = cooldown
		self.cooldown_counter = 0
		self.mode = mode
		self.threshold = threshold
		self.threshold_mode = threshold_mode
		self.best = None
		self.num_bad_epochs = None
		self.mode_worse = None  # the worse value for the chosen mode
		self.is_better = None
		self.eps = eps
		self.last_epoch = -1
		self._init_is_better(mode=mode, threshold=threshold,
							 threshold_mode=threshold_mode)
		self._reset()

	def _reset(self):
		"""Resets num_bad_epochs counter and cooldown counter."""
		self.best = self.mode_worse
		self.cooldown_counter = 0
		self.num_bad_epochs = 0

	def step(self, metrics, current_lr, epoch=None):
		current = metrics
		if epoch is None:
			epoch = self.last_epoch = self.last_epoch + 1
		self.last_epoch = epoch

		if self.is_better(current, self.best):
			self.best = current
			self.num_bad_epochs = 0
		else:
			self.num_bad_epochs += 1

		if self.in_cooldown:
			self.cooldown_counter -= 1
			self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

		if self.num_bad_epochs > self.patience:
			# self._reduce_lr(epoch)
			self.cooldown_counter = self.cooldown
			self.num_bad_epochs = 0
			current_lr = current_lr * self.factor
			logging.info('Update lr: %.8f' % current_lr)
			return current_lr

		return current_lr

	@property
	def in_cooldown(self):
		return self.cooldown_counter > 0

	def _cmp(self, mode, threshold_mode, threshold, a, best):
		if mode == 'min' and threshold_mode == 'rel':
			rel_epsilon = 1. - threshold
			return a < best * rel_epsilon

		elif mode == 'min' and threshold_mode == 'abs':
			return a < best - threshold

		elif mode == 'max' and threshold_mode == 'rel':
			rel_epsilon = threshold + 1.
			return a > best * rel_epsilon

		else:  # mode == 'max' and epsilon_mode == 'abs':
			return a > best + threshold

	def _init_is_better(self, mode, threshold, threshold_mode):
		if mode not in {'min', 'max'}:
			raise ValueError('mode ' + mode + ' is unknown!')
		if threshold_mode not in {'rel', 'abs'}:
			raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

		if mode == 'min':
			self.mode_worse = float('inf')
		else:  # mode == 'max':
			self.mode_worse = (-float('inf'))

		self.is_better = partial(self._cmp, mode, threshold_mode, threshold)