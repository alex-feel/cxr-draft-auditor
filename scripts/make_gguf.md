# Converting the merged MedGemma CXR model to GGUF (llama.cpp offline path)

> **Historical / reference document.** The live CXR Draft Auditor Space no longer uses GGUF or llama.cpp. Both models now run on the GPU (ZeroGPU) through Hugging Face transformers: the fine-tuned MedGemma grounds the image, and NVIDIA Nemotron-3 Nano 4B (BF16, `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`) parses the draft text. This guide is retained only as a record of the earlier offline llama.cpp experiment; the procedure below is no longer part of the serving path. The project does not claim a Llama Champion achievement.

This guide converts the merged 16-bit fine-tuned model (`your-username/medgemma-cxr-auditor`, produced by `train/train_medgemma_lora.py`) to GGUF for llama.cpp. It documents an offline / CPU inference path that does not depend on the ZeroGPU Space.

This was an exploratory offline path, never the serving path. The serving path is the merged bf16 model on the Gradio + ZeroGPU Space. Read the bounding-box fidelity caveat below before claiming the GGUF build matches cloud behavior.

## Bounding-box fidelity caveat (read first)

The auditor's value is image-grounded bounding boxes. Box-coordinate accuracy is sensitive to the vision tower's numeric precision and to how the multimodal projector is quantized. Quantizing the vision path aggressively (or down-quantizing the language model) can shift or blur the predicted boxes even when the textual labels still look right.

Therefore:

- Prefer `Q8_0` for the language model and keep the multimodal projector (`mmproj`) at `F16`. `Q8_0` is near-lossless for the text path; an `F16` projector preserves the visual feature precision the box coordinates depend on.
- Do NOT claim parity with the cloud bf16 model until you have validated grounded boxes from the GGUF build against the cloud model on a held-out set (per-finding IoU at 0.3 and 0.5, plus a visual spot check). Matching labels is not sufficient; the boxes must agree.
- If lower quants (`Q5_K_M`, `Q4_K_M`) are needed for size, treat them as degraded modes and re-run the box validation for each quant you publish. Document the measured box agreement per quant.

## What GGUF needs for a vision model

MedGemma 1.5 is a Gemma-3-family vision-language model. A vision GGUF is two files, not one:

1. The main model GGUF (the language model and embedded weights), quantized (for example `Q8_0`).
2. A separate multimodal projector file, `mmproj-*.gguf`, that carries the vision tower / projector. Keep this at `F16` per the caveat above.

llama.cpp loads both: the main GGUF with `-m`, the projector with `--mmproj`.

## Prerequisites (USER actions, not automated)

1. The merged 16-bit model exists on the Hub (or locally), for example `your-username/medgemma-cxr-auditor`. The model must be the verified merge from `train/train_medgemma_lora.py` (adapter confirmed folded in), not the bare adapter.
2. A working llama.cpp build with multimodal support and the Python conversion dependencies. Build tools (`build-essential`, `cmake`) must be installed before building.
3. A Hugging Face token if the merged model repo is private or gated (`hf auth login`).

## Step 1: Get the merged model locally

```bash
hf download your-username/medgemma-cxr-auditor --local-dir ./medgemma-cxr-merged
```

## Step 2: Build llama.cpp

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
pip install -r requirements.txt
cmake -B build -DGGML_CUDA=OFF
cmake --build build --target llama-quantize llama-mtmd-cli llama-server -j 4
```

`-DGGML_CUDA=OFF` is fine for conversion and CPU inference; enable CUDA only if you intend to offload layers to a local GPU at inference time.

## Step 3: Convert the language model to GGUF (F16, then quantize)

The standard converter emits the main model. For Gemma-3-family vision models it also writes the projector when the vision weights are present in the merged checkpoint.

```bash
# Main model to F16 GGUF.
python convert_hf_to_gguf.py ./medgemma-cxr-merged \
  --outfile medgemma-cxr-f16.gguf \
  --outtype f16

# Multimodal projector to F16 (keep the vision path at full precision).
python convert_hf_to_gguf.py ./medgemma-cxr-merged \
  --mmproj \
  --outfile mmproj-medgemma-cxr-f16.gguf
```

If your llama.cpp revision exposes the projector through a separate helper rather than the `--mmproj` flag, consult the current llama.cpp multimodal docs for the exact converter entry point; the goal is one main GGUF plus one `mmproj-*.gguf`.

## Step 4: Quantize the language model (keep the projector at F16)

```bash
./build/bin/llama-quantize medgemma-cxr-f16.gguf medgemma-cxr-q8_0.gguf Q8_0
```

Produce `Q8_0` as the recommended publishable quant. Generate `Q5_K_M` / `Q4_K_M` only if a smaller footprint is required, and re-validate boxes for each per the caveat. Do NOT quantize `mmproj-medgemma-cxr-f16.gguf`.

## Step 5: Run with llama-mtmd-cli (one-shot grounding)

`llama-mtmd-cli` is the multimodal CLI: it takes the main model, the projector, an image, and a prompt.

```bash
./build/bin/llama-mtmd-cli \
  -m medgemma-cxr-q8_0.gguf \
  --mmproj mmproj-medgemma-cxr-f16.gguf \
  --image /path/to/chest_xray.png \
  -p "$(cat your_pinned_image_grounding_prompt.txt)"
```

Use the project's pinned image-grounding prompt (the text returned by `cxr_auditor.prompts.build_image_grounding_prompt()`) so the GGUF model is prompted identically to the cloud path. The output is the constrained finding JSON list with normalized `box_2d` coordinates, parseable by `cxr_auditor.schema.extract_finding_list`.

## Step 6: Serve with llama-server (OpenAI-compatible, multimodal)

```bash
./build/bin/llama-server \
  -m medgemma-cxr-q8_0.gguf \
  --mmproj mmproj-medgemma-cxr-f16.gguf \
  --host 127.0.0.1 --port 8080
```

Send an image plus the pinned prompt through the OpenAI-compatible chat-completions endpoint (image as a `data:` URL in the message content). Add `-ngl <N>` to offload `N` layers to a local GPU if the build has CUDA/Metal enabled.

## Step 7: Validate boxes before publishing or claiming parity

Run the same held-out box evaluation used for the cloud model against the GGUF build (per-finding IoU at 0.3 and 0.5, plus a visual spot check on a handful of cases). Publish a GGUF quant only after its measured box agreement is documented. State the validated quant (`Q8_0` main + `F16` mmproj) and the measured box agreement in the GGUF repo card, and label any lower quant as a degraded offline mode rather than a cloud-equivalent one.
