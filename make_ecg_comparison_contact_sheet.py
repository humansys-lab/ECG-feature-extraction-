#!/usr/bin/env python3
"""Build a lossless-layout contact sheet from ECG comparison PNG files.

The source plots are only resized and placed on a white canvas; ECG curves,
markers, colors, and annotations are not regenerated or altered.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "ludb_ecgfeat_neurokit_many"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "contact_sheet_24_examples.png"


def _natural_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def discover_images(
    input_dir: Path,
    pattern: str = "*.png",
) -> list[Path]:
    return sorted(
        (
            path
            for path in input_dir.glob(pattern)
            if not path.name.startswith("contact_sheet")
        ),
        key=_natural_key,
    )


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def build_contact_sheet(
    image_paths: Sequence[Path],
    output_path: Path,
    *,
    columns: int = 4,
    tile_width: int = 700,
    tile_height: int = 520,
    caption_height: int = 42,
    margin: int = 18,
) -> Path:
    if not image_paths:
        raise ValueError("No source PNG files were found")
    if columns <= 0 or tile_width <= 0 or tile_height <= 0:
        raise ValueError("Grid dimensions must be positive")

    rows = (len(image_paths) + columns - 1) // columns
    cell_width = tile_width + 2 * margin
    cell_height = tile_height + caption_height + 2 * margin
    canvas = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(24)

    for index, image_path in enumerate(image_paths):
        row, column = divmod(index, columns)
        x = column * cell_width + margin
        y = row * cell_height + margin
        with Image.open(image_path) as source:
            source_rgb = source.convert("RGB")
            thumbnail = ImageOps.contain(
                source_rgb,
                (tile_width, tile_height),
                method=Image.Resampling.LANCZOS,
            )
        image_x = x + (tile_width - thumbnail.width) // 2
        image_y = y + (tile_height - thumbnail.height) // 2
        canvas.paste(thumbnail, (image_x, image_y))

        caption = image_path.stem.replace("_", " ")
        caption_y = y + tile_height + 7
        draw.text(
            (x + tile_width / 2, caption_y),
            caption,
            fill="#222222",
            font=font,
            anchor="ma",
        )
        draw.rectangle(
            (x - 1, y - 1, x + tile_width, y + tile_height),
            outline="#B8B8B8",
            width=1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a labeled contact sheet from ECG comparison PNGs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing individual comparison PNGs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output contact-sheet PNG.",
    )
    parser.add_argument(
        "--pattern",
        default="*.png",
        help=(
            "Glob pattern relative to --input-dir. It may include "
            "subdirectories, for example record_*/record_*_peaks.png."
        ),
    )
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=700)
    parser.add_argument("--tile-height", type=int, default=520)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    images = discover_images(
        args.input_dir.expanduser().resolve(),
        args.pattern,
    )
    output = build_contact_sheet(
        images,
        args.out.expanduser().resolve(),
        columns=args.columns,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
    )
    print(f"Images: {len(images)}")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
