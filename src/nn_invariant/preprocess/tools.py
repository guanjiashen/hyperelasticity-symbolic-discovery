
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def to_tensor(array: np.ndarray) -> torch.Tensor:
	return torch.from_numpy(array).float()


def split_mode_indices(
	sample_count: int,
	random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	indices = np.arange(sample_count)
	train_idx, temp_idx = train_test_split(
		indices,
		train_size=0.7,
		random_state=random_state,
		shuffle=True,
	)
	val_idx, test_idx = train_test_split(
		temp_idx,
		test_size=0.5,
		random_state=random_state,
		shuffle=True,
	)
	return train_idx, val_idx, test_idx