"""Verify real BF16 CUDA and cuDNN kernels before starting a research run."""
import json


def main():
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 CUDA device required')
    capability = torch.cuda.get_device_capability(0)
    if capability >= (12, 0) and torch.version.cuda not in ('12.8', '12.9', '13.0', '13.1'):
        raise RuntimeError('Blackwell requires a supported CUDA build; use requirements-v15-cuda.lock')
    with torch.inference_mode():
        value = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
        result = value @ value
        assert bool(torch.all(result == 16).item())
        # Official audio tokenizers include recurrent modules even for text-only loading.
        recurrent = torch.nn.LSTM(8, 8, batch_first=True).cuda()
        sequence, _ = recurrent(torch.ones((1, 2, 8), device='cuda'))
        assert bool(torch.isfinite(sequence).all().item())
        torch.cuda.synchronize()
    print('V15_REAL_CUDA_KERNELS_VERIFIED ' + json.dumps({
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'capability': capability,
        'bf16_matmul': True, 'cudnn_recurrent_loading': True}), flush=True)


if __name__ == '__main__':
    main()
