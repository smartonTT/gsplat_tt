import argparse

from backends import REGISTRY as BACKEND_REGISTRY
from gsplat.loading_gaussians import load_ply
from gsplat.viewer import GaussianViewer


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D Gaussian Splatting Viewer"
    )
    parser.add_argument("ply_path", help="Path to a pre-trained .ply file")
    parser.add_argument(
        "--port", type=int, default=8080, help="Viewer port (default: 8080)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Viewer host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--backend",
        choices=sorted(BACKEND_REGISTRY),
        default="cpu",
        help=(
            "Rendering backend; choices come from the registry in "
            "backends/__init__.py. Default: cpu."
        ),
    )
    parser.add_argument(
        "--render-width",
        type=int,
        default=1024,
        help="Initial render width in pixels (snapped to multiples of 32). Default: 1024.",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=1024,
        help="Initial render height in pixels (snapped to multiples of 32). Default: 1024.",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=None,
        help=(
            "Deprecated alias: if set, overrides --render-width and --render-height "
            "with a square size."
        ),
    )
    parser.add_argument(
        "--force-square",
        type=int,
        default=None,
        help=(
            "If set, every frame renders at exactly NxN regardless of the "
            "Render Res UI (must be a multiple of 32). Overrides render width/height."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-frame stage timing.",
    )
    args = parser.parse_args()

    print(f"Loading Gaussians from {args.ply_path}...")
    gaussians = load_ply(args.ply_path)
    print(f"Loaded {gaussians.num_gaussians:,} Gaussians")

    viewer = GaussianViewer(
        gaussians,
        host=args.host,
        port=args.port,
        backend=args.backend,
        render_width=args.render_width,
        render_height=args.render_height,
        max_resolution=args.max_resolution,
        force_square=args.force_square,
        verbose=args.verbose,
        scene_path=args.ply_path,
    )
    viewer.run()


if __name__ == "__main__":
    main()
