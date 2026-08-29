# Example assets

This folder holds the chest X-ray images referenced by the `gr.Examples` block in `app.py`. Three example images are committed:

- `cxr_example_1.png`, `cxr_example_2.png`, `cxr_example_3.png` - frontal chest X-rays from NIH ChestX-ray14 (see Source and licensing), used to drive the examples panel. Each is paired with a draft impression so the examples demonstrate three audit outcomes the tool exists to catch: a clean draft (no discrepancy), a draft that misses a finding the image shows, and a draft that over-calls a finding the image does not support.
- `cxr_example_4.png` - a chest X-ray showing a pulmonary mass, paired with a falsely-clean draft so the example demonstrates an URGENT missed finding (nodule / mass is on the urgent whitelist). This image has a different source and license from the NIH examples; see Source and licensing.

## Source and licensing

`cxr_example_1.png` through `cxr_example_3.png` are frontal chest X-rays from the NIH ChestX-ray14 dataset (NIH Clinical Center), which the provider releases with no restrictions on use, so they are safe to redistribute in this public Space. They are sourced here from the ChestX-Det redistribution (`natealberti/ChestX-Det`, Apache-2.0 annotations over the same NIH images). Citation for the underlying images: Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases," IEEE CVPR 2017.

`cxr_example_4.png` (the urgent mass example) has a different source: "Lung cancer as seen on chest radiograph" by James Heilman, MD, via Wikimedia Commons (<https://commons.wikimedia.org/wiki/File:LungCACXR.PNG>), licensed CC BY-SA 3.0. It is redistributed here unmodified under that license with attribution; CC BY-SA 3.0 also requires share-alike on any altered or derivative version of the image. The CC BY-SA 3.0 license on this one image is independent of the Apache-2.0 license on the Space's own application code.

Do NOT commit images from a source whose license forbids redistribution: VinDr-CXR is under a non-commercial research Data Use Agreement (not CC0), and IU-Xray / Open-i images are CC BY-NC-ND. To use different examples, drop chest X-rays you are licensed to redistribute into this directory.

## How the examples are wired

`app.py` builds the example paths with `_example_path(...)`, resolved relative to this `examples/` directory, and registers only example image files that are actually present (so the panel appears once the images exist and stays hidden otherwise). If you use different file names, update the `candidate_examples` list in `app.py` to match.

On ZeroGPU, `gr.Examples` caching defaults to lazy (the example runs on first click, not at startup, because no GPU is attached at startup). If you replace an example image in place, rename the file so the path-keyed cache does not serve a stale cached output.
