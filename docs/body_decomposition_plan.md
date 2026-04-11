# Body Decomposition Implementation Plan (v3)

## What failed

### v1: Normal-Axis Partitioning
Split faces by normal direction → grouped "planes vs cylinders" instead of physical bodies.

### v2: Cross-Section Sweep + Centroid Assignment
Found the correct split PLANE (Z=-96 where area jumps 10x) but couldn't cleanly assign
faces to bodies. Centroid-based assignment creates diagonal ramps. Max-vertex creates
bleeding. Flood-fill from caps bleeds through the connected mesh topology.

## The fundamental problem
The mesh is a single connected watertight surface. There are NO topological boundaries
between the "plate" and the "bracket". Every face is connected to every other face
through shared edges. Any flood-fill or connectivity-based approach will treat it as
one body.

## What would actually work

### Approach: Virtual Planar Cut
1. Find the split plane (cross-section sweep already works — score 10.3 at Z=-96)
2. DON'T try to assign existing faces to bodies
3. Instead, SLICE the mesh with a cutting plane at Z=-96
4. This creates two separate watertight sub-meshes
5. Each sub-mesh IS a body — topologically disconnected

### Implementation
```python
import trimesh

# Find split plane via cross-section sweep (already implemented)
split_axis = 2  # Z
split_pos = -96.0

# Cut the mesh into two pieces
plane_origin = [0, 0, split_pos]
plane_normal = [0, 0, 1]

# trimesh.intersections.slice_mesh_plane returns the portion ABOVE the plane
body_above = trimesh.intersections.slice_mesh_plane(mesh, plane_normal, plane_origin)
body_below = trimesh.intersections.slice_mesh_plane(mesh, [-n for n in plane_normal], plane_origin)
```

### Why this works
- The cut creates a NEW face at the split plane (fills the hole)
- Each piece is a valid watertight sub-mesh
- No face assignment ambiguity — every face is in exactly one piece
- The cut is geometrically clean — a perfect plane

### Challenges
- Need to map cut faces back to original mesh for visualization coloring
- The new faces at the cut plane need normals
- trimesh's slice_mesh_plane may not produce watertight results for complex geometry
- Performance: slicing is fast for simple cuts

### Visualization
For the decomposition preview, DON'T color original mesh faces.
Instead, render the two cut pieces with different materials/colors,
slightly separated (exploded view) so the user can see the split.

### Per-body pipeline
Each cut piece is a standalone CleanedMesh. Stages 2-5 run on each piece.
Stage 6 builds each piece as a separate extrusion and unions them.

## Priority
This is a research task. The single-body pipeline works well for most parts.
Multi-body decomposition should be a V2 feature.
