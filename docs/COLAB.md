# Colab and mounted-drive restart behavior

Colab cannot guarantee background execution after the browser disconnects or the
runtime is reclaimed. ParaSeedBench can resume committed work, but it cannot keep a
terminated runtime alive or automatically recreate the same GPU/software stack.

Before the first generation, mount Drive and set `output_dir` in a new configuration
to an absolute Drive path. Store the locked configuration and this exact source
release in Drive as well. Run one writer only.

After reconnection:

1. mount the same Drive;
2. restore the same source and Python/CUDA package versions;
3. activate/recreate the environment;
4. confirm that no old writer exists;
5. clear a stale lock only if needed;
6. rerun the identical locked command.

Validated images, per-image score JSONs, and embedding chunks are reused. The image
whose denoising was interrupted is recomputed. Wait for Drive synchronization before
ending a runtime; local atomic writes do not prove remote synchronization.

For a multi-hour main experiment, the RTX 6000 Ada workstation is preferable because
its runtime and GPU identity remain stable.
