# Upload this release to GitHub

1. Extract the release ZIP.
2. Replace `LICENSE_NOTICE.md` with the license you choose.
3. Create an empty repository on GitHub without adding a README or `.gitignore`.
4. From the extracted `ParaSeedBench` directory:

```bash
git init
git add .
git commit -m "Release ParaSeedBench v0.3.0"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/ParaSeedBench.git
git push -u origin main
git tag -a v0.3.0 -m "Paper-aligned ParaSeedBench release"
git push origin v0.3.0
```

Do not upload `.venv`, Hugging Face caches, `outputs/`, generated images, or model
weights. They are ignored by `.gitignore`. The released numerical CSVs and labels
under `paper_artifacts/` are small enough for ordinary Git; Git LFS is not needed.

After workstation access returns, add the original run manifest as described in
`REPRODUCIBILITY.md`, run both verification commands, and commit it as a provenance
update. Do not commit a private access token or user-specific cache path.
