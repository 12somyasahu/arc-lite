# ARC-Lite

An original 11M parameter decoder-only transformer trained on ARC-AGI-1 using an MDL-inspired joint loss over input and output grids. Built from scratch on consumer hardware.

## Inspiration & Credits

Conceptually inspired by [Mithil Vakde's mdlARC](https://github.com/mvakde/mdlARC) — specifically the core idea of training on both input and output grids jointly using the MDL principle. Every line of code in ARC-Lite is original.

## Architecture

- Decoder-only transformer, 11.186M parameters
- d_model=384, n_heads=8, n_layers=6, d_ff=1536
- 2D spatial positional embeddings (row + col) for grid-aware attention
- Memory-efficient attention via `torch.nn.functional.scaled_dot_product_attention`
- Weight-tied LM head (GPT-2 style)
- GPT-2 style weight initialisation with residual scaling

## MDL-Inspired Training Objective

Standard ARC solvers train only on output grid prediction. ARC-Lite trains on the **entire sequence** — input grid + output grid jointly — as a next-token prediction task:

```
[BOS] input_grid_tokens [SEP] output_grid_tokens [EOS]
```

This forces the model to compress and reconstruct both grids, learning the transformation implicitly.

## Token Vocabulary (15 tokens)

| Token | ID |
|-------|----|
| Colors 0-9 | 0-9 |
| PAD | 10 |
| BOS | 11 |
| EOS | 12 |
| SEP | 13 |
| ROW | 14 |

## Hardware

- Training: NVIDIA RTX 3050 6GB Laptop GPU (smoke runs) + Kaggle T4 x2 (full run)
- 20,000 steps, effective batch size 64 (16 × 4 grad accum)
- Warmup 500 steps + cosine decay to 3e-5

## Results

| Split | Loss | PPL |
|-------|------|-----|
| Train (step 500) | 1.17 | 3.24 |
| Eval (step 500) | 1.27 | 3.57 |
| Final eval accuracy | TBD |

## Project Structure

```
arc-lite/
├── src/
│   ├── data/          # Dataset loader, downloader, augmentations
│   ├── model/         # Transformer, attention, embeddings, tokenizer
│   ├── training/      # Trainer, MDL loss, LR scheduler
│   └── evaluation/    # Exact-match ARC evaluator
├── scripts/
│   ├── train.py       # Training entry point
│   └── evaluate.py    # Evaluation entry point
├── api/               # FastAPI inference server (coming soon)
└── kaggle_train.ipynb # Kaggle training notebook
```

## Setup

```bash
git clone https://github.com/12somyasahu/arc-lite.git
cd arc-lite
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python src/data/download.py  # Downloads ARC-AGI-1 dataset
```

## Train

```bash
python scripts/train.py --max_steps 20000
```

## Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/arc_lite_best.pt
```

## License

MIT
