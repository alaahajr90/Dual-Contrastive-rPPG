"""Parameter and model-only latency profiling; never includes video or preprocessing time."""
import time
import torch


def profile_model(model, device=None, warmup=10, iterations=50):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    dummy = torch.zeros(1, 3, 128, 64, 64, device=device)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(warmup):
            model(dummy)
        sync()
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(dummy)
            sync()
            samples.append((time.perf_counter() - start) * 1000)
    return {"total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "mean_forward_ms": sum(samples) / len(samples), "device": str(device),
            "iterations": iterations, "excludes_preprocessing": True}
