import argparse
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ut_project.models import UNetTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a Mermaid model graph and layer summary.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "model_graph.md")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def format_shape(value):
    if isinstance(value, torch.Tensor):
        return "x".join(str(dim) for dim in value.shape)
    if isinstance(value, (list, tuple)):
        return ", ".join(format_shape(item) for item in value)
    return str(type(value).__name__)


def is_leaf_module(module: nn.Module):
    return len(list(module.children())) == 0


def count_parameters(module: nn.Module):
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def collect_layer_summary(model: nn.Module, sample_input: torch.Tensor):
    rows = []
    hooks = []

    def hook_fn(name, module):
        def _hook(_module, inputs, output):
            rows.append(
                {
                    "name": name,
                    "type": module.__class__.__name__,
                    "input_shape": format_shape(inputs),
                    "output_shape": format_shape(output),
                    "params": count_parameters(module),
                }
            )

        return _hook

    for name, module in model.named_modules():
        if name and is_leaf_module(module):
            hooks.append(module.register_forward_hook(hook_fn(name, module)))

    model.eval()
    with torch.no_grad():
        output = model(sample_input)

    for hook in hooks:
        hook.remove()

    return rows, output


def build_mermaid(base_channels: int, embed_dim: int, height: int, width: int):
    h1, w1 = height, width
    h2, w2 = height // 2, width // 2
    h3, w3 = height // 4, width // 4
    h4, w4 = height // 8, width // 8
    tokens = h4 * w4
    c = base_channels

    return f"""```mermaid
flowchart TD
    A["Input noisy image<br/>1x3x{h1}x{w1}"] --> B["Stem ConvBlock<br/>1x{c}x{h1}x{w1}"]
    B --> C["Down1 MaxPool + ConvBlock<br/>1x{c * 2}x{h2}x{w2}"]
    C --> D["Down2 MaxPool + ConvBlock<br/>1x{c * 4}x{h3}x{w3}"]
    D --> E["Down3 MaxPool + ConvBlock<br/>1x{c * 8}x{h4}x{w4}"]
    E --> F["Bottleneck ConvBlock<br/>1x{c * 8}x{h4}x{w4}"]
    F --> G["1x1 projection<br/>1x{embed_dim}x{h4}x{w4}"]
    G --> H["Flatten tokens<br/>1x{tokens}x{embed_dim}"]
    H --> I["Transformer Encoder"]
    I --> J["Reshape + 1x1 projection<br/>1x{c * 8}x{h4}x{w4}"]
    F --> K["Residual add"]
    J --> K
    K --> L["Up1 + skip Down2<br/>1x{c * 4}x{h3}x{w3}"]
    D --> L
    L --> M["Up2 + skip Down1<br/>1x{c * 2}x{h2}x{w2}"]
    C --> M
    M --> N["Up3 + skip Stem<br/>1x{c}x{h1}x{w1}"]
    B --> N
    N --> O["Head 1x1 Conv<br/>predicted noise 1x3x{h1}x{w1}"]
    A --> P["Denoised image = noisy - predicted noise"]
    O --> P
```"""


def render_markdown(rows, output, args):
    total_params = sum(row["params"] for row in rows)
    lines = [
        "# UNet Transformer Model Graph",
        "",
        f"Input shape: `1x3x{args.height}x{args.width}`",
        f"Output shape: `{format_shape(output)}`",
        f"Leaf-module parameters: `{total_params:,}`",
        "",
        build_mermaid(args.base_channels, args.embed_dim, args.height, args.width),
        "",
        "## Layer Summary",
        "",
        "| # | Module | Type | Input | Output | Params |",
        "|---:|---|---|---|---|---:|",
    ]

    for index, row in enumerate(rows, start=1):
        lines.append(
            f"| {index} | `{row['name']}` | `{row['type']}` | "
            f"`{row['input_shape']}` | `{row['output_shape']}` | {row['params']:,} |"
        )

    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    model = UNetTransformer(
        base_channels=args.base_channels,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        transformer_layers=args.transformer_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    sample_input = torch.zeros(1, 3, args.height, args.width)
    rows, output = collect_layer_summary(model, sample_input)
    markdown = render_markdown(rows, output, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"saved model graph to: {args.output}")
    print(f"output shape: {format_shape(output)}")


if __name__ == "__main__":
    main()
