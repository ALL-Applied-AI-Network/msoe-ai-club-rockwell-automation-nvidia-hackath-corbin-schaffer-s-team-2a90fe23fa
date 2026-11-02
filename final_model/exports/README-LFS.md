# Restore the standalone GB10 executable

GitHub limits this repository to 2 GiB per LFS file. The executable is stored losslessly in two LFS parts. After cloning:

```bash
git lfs pull
sh final_model/exports/restore-vial-inspector-gb10.sh
./final_model/exports/vial-inspector-gb10 --help
```

The script reconstructs the executable, verifies its original SHA256, and makes it executable. No Python is required for reconstruction. The reconstructed file is ignored by Git; the two parts are tracked by LFS. Keep about 3 GB free for reconstruction in addition to the downloaded parts. This executable targets Linux ARM64 NVIDIA GB10 with a compatible NVIDIA driver; splitting does not change device compatibility. Details and individual part hashes are in vial-inspector-gb10.parts.json.
