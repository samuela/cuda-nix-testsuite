# cuda-nix-testsuite

This is a suite of tests for CUDA-enabled packages in Nixpkgs. The Nix build environment does not support GPU-access by default. So we have to do things off-the-books!

TODO:

- [ ] A simple runner script that scrapes all scripts that look like `*.test.*` and runs them.
- [ ] Add support for tests in subdirectories with `shell.nix` instead of nix-shell shebang-lines.
- [ ] Add test for basic PyTorch functionality
- [ ] Add test for [jax Conv issue](https://github.com/google/jax/discussions/9455)
- [ ] Add test for basic TF functionality
- [ ] Add test for https://github.com/NixOS/nixpkgs/issues/163739
- [ ] Add test for https://github.com/NixOS/nixpkgs/pull/153542#issuecomment-1066180952

## Development

### How do I run the test suite?

TODO: there will be a runner script (coming soon!)

### How do I add a test?

Create a new script in the `tests/` subdirectory with `.test.` in the filename.
See [jax.test.py](https://github.com/samuela/cuda-nix-testsuite/blob/main/tests/jax.test.py) for an example.
