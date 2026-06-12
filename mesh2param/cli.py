"""mesh2param command-line interface.

Examples:
    mesh2param convert input.stl --dry-run --output feature_tree.json
    mesh2param convert input.stl --step output.step
    mesh2param convert input.stl --onshape-doc "My Document" --onshape-workspace main
"""

import json
import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """Convert triangulated meshes of prismatic parts into parametric CAD."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, default=False,
              help="Plan the Onshape export without calling the API.")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Write the feature tree + export plan as JSON.")
@click.option("--step", "step_path", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Export a STEP file via CadQuery (offline path).")
@click.option("--onshape-doc", default=None, help="Onshape document name for live export.")
@click.option("--onshape-workspace", default="main", help="Onshape workspace name.")
@click.option("--remesh/--no-remesh", default=False,
              help="Remesh during ingest (off by default to match server output).")
def convert(input_file: Path, dry_run: bool, output: Path | None, step_path: Path | None,
            onshape_doc: str | None, onshape_workspace: str, remesh: bool) -> None:
    """Run the full pipeline on INPUT_FILE (STL/OBJ/PLY/3MF)."""
    from mesh2param.config import (
        ExportConfig, FitConfig, IngestConfig, RecognizeConfig, SegmentConfig, TopologyConfig,
    )
    from mesh2param.pipeline.export_onshape import export_to_onshape
    from mesh2param.pipeline.fit import fit_surfaces
    from mesh2param.pipeline.ingest import ingest_mesh
    from mesh2param.pipeline.recognize import recognize_features
    from mesh2param.pipeline.segment import segment_mesh
    from mesh2param.pipeline.topology import build_topology

    click.echo(f"Stage 1/6: ingesting {input_file.name} ...")
    cleaned = ingest_mesh(str(input_file), IngestConfig(remesh=remesh))
    click.echo(f"  {cleaned.num_faces} faces, {cleaned.num_vertices} vertices, "
               f"watertight={cleaned.is_watertight}")

    click.echo("Stage 2/6: segmenting surfaces ...")
    regions = segment_mesh(cleaned, SegmentConfig())
    click.echo(f"  {len(regions)} regions")

    click.echo("Stage 3/6: fitting primitives ...")
    fitted = fit_surfaces(regions, cleaned, FitConfig())

    click.echo("Stage 4/6: building topology ...")
    graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())

    click.echo("Stage 5/6: recognizing features ...")
    feature_tree = recognize_features(regions, fitted, graph, edges, cleaned, RecognizeConfig())
    click.echo(f"  {feature_tree.num_features} features")

    if step_path is not None:
        click.echo(f"Stage 6/6: exporting STEP via CadQuery -> {step_path} ...")
        from mesh2param.pipeline.export_cadquery import export_cadquery
        cq_result = export_cadquery(feature_tree, cleaned, output_path=str(step_path))
        status = getattr(cq_result, "status", "done")
        click.echo(f"  CadQuery export status: {status}")

    effective_dry_run = dry_run or onshape_doc is None
    if onshape_doc and not dry_run:
        click.echo(f"Stage 6/6: exporting to Onshape doc {onshape_doc!r} "
                   f"(workspace {onshape_workspace!r}) ...")
    else:
        click.echo("Stage 6/6: planning Onshape export (dry run) ...")
    result = export_to_onshape(feature_tree, cleaned, ExportConfig(
        dry_run=effective_dry_run, document_name=onshape_doc or "",
    ))
    click.echo(f"  status={result.status}, features={result.feature_count}, "
               f"volume_error={result.volume_error_pct:.2f}%")
    if result.document_url and not effective_dry_run:
        click.echo(f"  document: {result.document_url}")

    if output is not None:
        payload = {
            "input": str(input_file),
            "feature_tree": feature_tree.to_dict(),
            "export": result.to_dict(),
        }
        output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        click.echo(f"Wrote {output}")

    if str(result.status).upper() in ("FAIL", "ERROR"):
        sys.exit(1)


if __name__ == "__main__":
    main()
