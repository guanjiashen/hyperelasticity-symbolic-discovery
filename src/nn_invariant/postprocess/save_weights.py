
import json
from pathlib import Path

import torch
import torch.nn as nn

def save_weights(
	model: nn.Module,
	base_dir: Path,
	model_name: str = 'ffbp_model.pt',
	weights_name: str = 'ffbp_weights.txt',
	) -> None:
	base_dir = base_dir.resolve()
	base_dir.mkdir(parents=True, exist_ok=True)

	model_path = base_dir / model_name
	weights_path = base_dir / weights_name
	metadata_path = model_path.with_suffix('.json')
	hidden_layers = [int(layer.out_features) for layer in (model.hidden, *model.additional_hidden)]

	torch.save(model.state_dict(), model_path)
	metadata_path.write_text(
		json.dumps(
			{
				'hidden_neurons': hidden_layers[0],
				'hidden_layers': hidden_layers,
				'activation': str(getattr(model, 'activation_name', 'softplus')),
			},
			indent=2,
		) + '\n',
		encoding='utf-8',
	)
		
	model_path = model_path.resolve()
	weights_path = weights_path.resolve()
	state_dict = torch.load(model_path, map_location='cpu')    
	
	hidden_weights = [state_dict['hidden.weight'].detach().cpu().numpy()]
	hidden_biases = [state_dict['hidden.bias'].detach().cpu().numpy()]
	for index in range(len(hidden_layers) - 1):
		hidden_weights.append(state_dict[f'additional_hidden.{index}.weight'].detach().cpu().numpy())
		hidden_biases.append(state_dict[f'additional_hidden.{index}.bias'].detach().cpu().numpy())
	output_weight = state_dict['output.weight'].detach().cpu().numpy().reshape(-1)
	output_bias = float(state_dict['output.bias'].detach().cpu().numpy().reshape(-1)[0])

	_, input_dim = hidden_weights[0].shape
	if input_dim != 2:
		raise ValueError(f'Expected 2 network inputs, got {input_dim}.')

	weights_path.parent.mkdir(parents=True, exist_ok=True)
	lines = ['# FFBP constitutive model weights']
	if len(hidden_layers) == 1:
		lines.extend(['hidden_neurons', str(hidden_layers[0])])
	else:
		lines.extend(['hidden_layers', ' '.join(str(width) for width in hidden_layers)])
	for index, (weight, bias) in enumerate(zip(hidden_weights, hidden_biases)):
		weight_label = 'hidden_weights' if len(hidden_layers) == 1 else f'hidden_{index}_weights'
		bias_label = 'hidden_bias' if len(hidden_layers) == 1 else f'hidden_{index}_bias'
		lines.append(weight_label)
		for row in weight:
			lines.append(' '.join(f'{float(value):.17g}' for value in row))
		lines.append(bias_label)
		for value in bias:
			lines.append(f'{float(value):.17g}')

	lines.append('output_weights')
	for value in output_weight:
		lines.append(f'{float(value):.17g}')

	lines.append('output_bias')
	lines.append(f'{output_bias:.17g}')

	weights_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

	print(f'Saved {model_path.name}, {metadata_path.name}, and {weights_path.name}')
