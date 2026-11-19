import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .model import TSK_SPCN
else:
    from model import TSK_SPCN


PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_NAMES = ('x1_mapped', 'x2_mapped', 'risk_score1', 'risk_score2', 'strengths2')


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Expected a positive integer.')
    return number


def load_pretrained(dataset):
    manifest = json.loads((PACKAGE_DIR / 'weights' / 'manifest.json').read_text(encoding='utf-8'))
    record = manifest['checkpoints'][dataset]
    path = PACKAGE_DIR / 'weights' / record['filename']
    if not path.is_file():
        raise FileNotFoundError(f'Bundled checkpoint is missing: {path}')
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Checkpoint checksum differs from the bundled manifest.')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format_version') != 1 or saved.get('model_name') != 'TSK-SPCN':
        raise ValueError('Unsupported checkpoint format.')
    for key in ('dataset', 'seed', 'n_features', 'dtype', 'model_config', 'selected_epoch'):
        if saved[key] != record[key]:
            raise ValueError(f'Checkpoint metadata mismatch: {key}')
    dtype = {'float32': torch.float32, 'float64': torch.float64}[saved['dtype']]
    model = TSK_SPCN(saved['n_features'], **saved['model_config']).to(dtype=dtype)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.eval()
    return model, saved, record


def prepare_input(path, features, window, samples, seed):
    if path is None:
        values = np.random.default_rng(seed).random((samples, features, window))
        return values, 'synthetic_smoke_test_only'
    path = path.expanduser().resolve()
    if path.suffix.lower() != '.npy':
        raise ValueError('Input must be a .npy array of already prepared windows (B,F,W).')
    values = np.load(path, allow_pickle=False)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1:] != (features, window):
        raise ValueError(f'Expected nonempty input (B,{features},{window}); got {values.shape}. No automatic windowing is performed.')
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError('Input must contain real numeric values.')
    if not np.isfinite(values).all():
        raise ValueError('Input contains NaN or infinity.')
    return values, 'user_prepared_windows'


def forward_batches(model, values, batch_size, device):
    model = model.to(device=device)
    dtype = next(model.parameters()).dtype
    outputs = [[] for _ in OUTPUT_NAMES]
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            inputs = torch.as_tensor(values[start:start + batch_size], dtype=dtype, device=device)
            if not torch.isfinite(inputs).all():
                raise ValueError('Input cannot be represented as finite values in the model dtype.')
            result = model(inputs)
            if len(result) != len(OUTPUT_NAMES):
                raise ValueError('Unexpected model output count.')
            for name, destination, tensor in zip(OUTPUT_NAMES, outputs, result):
                if not torch.isfinite(tensor).all():
                    raise ValueError(f'Nonfinite model output: {name}')
                destination.append(tensor.cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in zip(OUTPUT_NAMES, outputs)}


def main():
    parser = argparse.ArgumentParser(description='Load bundled pretrained weights and run model forward only; no anomaly scoring or benchmark evaluation.')
    parser.add_argument('--dataset', choices=('GWD', 'SWaT', 'wave2022'), default='GWD')
    parser.add_argument('--input', type=Path, help='Prepared .npy windows (B,F,W); omitted: synthetic forward smoke test.')
    parser.add_argument('--output', type=Path, help='Output .npz; defaults to artifacts/<dataset>_forward.npz inside this package.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=positive_integer, default=32)
    parser.add_argument('--samples', type=positive_integer, default=4, help='Synthetic sample count when --input is omitted.')
    parser.add_argument('--seed', type=int, default=53, help='Only controls synthetic input and forward-test RNG; does not select or retrain weights.')
    args = parser.parse_args()
    dataset = 'GWD' if args.dataset == 'wave2022' else args.dataset
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu.')
    torch.manual_seed(args.seed)
    model, saved, record = load_pretrained(dataset)
    values, input_kind = prepare_input(args.input, saved['n_features'], saved['model_config']['n_window'], args.samples, args.seed)
    outputs = forward_batches(model, values, args.batch_size, device)
    output = args.output.expanduser().resolve() if args.output is not None else PACKAGE_DIR / 'artifacts' / f'{dataset}_forward.npz'
    if output.suffix.lower() != '.npz':
        parser.error('--output must have a .npz extension')
    metadata = {
        'dataset': dataset,
        'source_dataset': saved['source_dataset'],
        'checkpoint_seed': saved['seed'],
        'selected_epoch_zero_based': saved['selected_epoch'],
        'checkpoint_sha256': record['sha256'],
        'model_config': saved['model_config'],
        'dtype': saved['dtype'],
        'device': str(device),
        'input_kind': input_kind,
        'input_shape': list(values.shape),
        'forward_test_seed': args.seed,
        'anomaly_scores_or_benchmark_metrics_computed': False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, **outputs, metadata_json=np.asarray(json.dumps(metadata)))
    print(json.dumps({**metadata, 'output': str(output), 'output_shapes': {key: list(value.shape) for key, value in outputs.items()}, 'note': 'Model forward outputs only. Synthetic inputs do not measure anomaly detection performance. Existing output files are never overwritten.'}, indent=2))


if __name__ == '__main__':
    main()
