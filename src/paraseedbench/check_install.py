from __future__ import annotations


def main() -> None:
    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"VRAM: {props.total_memory / 2**30:.1f} GiB")
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        print(f"FP16 smoke sum: {float((x @ x).abs().mean()):.4f}")


if __name__ == "__main__":
    main()
