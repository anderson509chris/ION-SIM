"""Electrode geometry from CAD files.

Each solid body in the file becomes one electrode, returned as a triangle mesh that
Grid3D.add_mesh voxelises.
  * STEP / IGES / BREP: read with gmsh (OpenCASCADE), surfaces meshed, one body per
    volume (or per surface if the file has no volumes). Units are always mm.
  * STL / OBJ / PLY / OFF / GLB: read with trimesh. Named objects (OBJ "o"/"g", scene
    nodes) become bodies; a single mesh is split into its connected pieces. These
    formats have no units, so the caller says which unit the numbers are in.
"""
import os
import numpy as np

CAD_EXT = (".step", ".stp", ".iges", ".igs", ".brep")
MESH_EXT = (".stl", ".obj", ".ply", ".off", ".glb", ".gltf")
UNITS = {"mm": 1e-3, "m": 1.0, "cm": 1e-2, "µm": 1e-6, "inch": 25.4e-3}
FILE_FILTER = "Geometry (*.step *.stp *.iges *.igs *.brep *.stl *.obj *.ply *.off *.glb *.gltf)"


def load_bodies(path, units="mm", axis="z"):
    """Returns [(name, triangles (M,3,3) in metres)]. `axis` is the CAD axis the ions travel
    along; coordinates are permuted so that it becomes z."""
    ext = os.path.splitext(path)[1].lower()
    if ext in CAD_EXT:
        bodies = _load_occ(path)
        scale = 1e-3
    elif ext in MESH_EXT:
        bodies = _load_mesh(path)
        scale = UNITS[units]
    else:
        raise ValueError(f"Unsupported file type {ext}")
    perm = {"x": [1, 2, 0], "y": [2, 0, 1], "z": [0, 1, 2]}[axis]
    out = [(name, tris[:, :, perm] * scale) for name, tris in bodies if len(tris)]
    return _number_unnamed(out)


def _number_unnamed(bodies):
    """Bodies the file left unnamed ("solid 7") are renumbered in order along the ion
    axis, so "solid 1" is the first part the ions meet. Named parts keep their names."""
    import re
    unnamed = [k for k, (n, _) in enumerate(bodies) if re.fullmatch(r"(solid|surface) \d+( \(\d+\))?", n)]
    order = sorted(unnamed, key=lambda k: float(bodies[k][1][:, :, 2].mean()))
    out = list(bodies)
    for rank, k in enumerate(order, start=1):
        kind = bodies[k][0].split()[0]
        out[k] = (f"{kind} {rank}", bodies[k][1])
    return out


def _load_occ(path):
    import gmsh
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.model.add("cad")
        gmsh.model.occ.importShapes(path)
        gmsh.model.occ.synchronize()
        bb = np.array(gmsh.model.getBoundingBox(-1, -1))
        size = np.linalg.norm(bb[3:] - bb[:3])
        gmsh.option.setNumber("Mesh.MeshSizeMax", size / 40)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
        gmsh.model.mesh.generate(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        xyz = np.zeros((int(tags.max()) + 1, 3))
        xyz[tags.astype(int)] = coords.reshape(-1, 3)

        def surface_tris(stag):
            types, _, nodes = gmsh.model.mesh.getElements(2, stag)
            out = [xyz[n.astype(int).reshape(-1, 3)] for t, n in zip(types, nodes) if t == 2]
            return np.concatenate(out) if out else np.zeros((0, 3, 3))

        def label(dim, tag, k, kind):
            # gmsh names look like "Shapes/<assembly>/<part>"; keep the part name
            name = gmsh.model.getEntityName(dim, tag).split("/")[-1].strip()
            return name if name and "translator" not in name.lower() else f"{kind} {k}"

        bodies = []
        vols = gmsh.model.getEntities(3)
        if vols:
            for k, (_, vtag) in enumerate(vols, start=1):
                surfs = gmsh.model.getBoundary([(3, vtag)], oriented=False, combined=False)
                tris = np.concatenate([surface_tris(abs(s)) for _, s in surfs])
                bodies.append((label(3, vtag, k, "solid"), tris))
        else:
            for k, (_, stag) in enumerate(gmsh.model.getEntities(2), start=1):
                bodies.append((label(2, stag, k, "surface"), surface_tris(stag)))
        names = [n for n, _ in bodies]                     # make duplicate names unique
        return [(n if names.count(n) == 1 else f"{n} ({k})", t) for k, (n, t) in enumerate(bodies, start=1)]
    finally:
        gmsh.finalize()


def _load_obj(path):
    """OBJ reader that keeps each "o"/"g" object as its own body (trimesh merges objects
    sharing a material). Polygons are fanned into triangles."""
    verts, bodies, name, faces = [], [], None, []

    def close():
        if faces:
            bodies.append((name or f"body {len(bodies) + 1}", np.array(faces)))

    with open(path, errors="replace") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(c) for c in parts[1:4]])
            elif parts[0] in ("o", "g") and len(parts) > 1:
                close()
                name, faces = " ".join(parts[1:]), []
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) for p in parts[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                faces += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    close()
    v = np.array(verts)
    return [(n, v[f]) for n, f in bodies]


def _load_mesh(path):
    import trimesh
    stem = os.path.splitext(os.path.basename(path))[0]
    if path.lower().endswith(".obj"):
        bodies = _load_obj(path)
        if len(bodies) != 1:
            return bodies
    obj = trimesh.load(path)
    if isinstance(obj, trimesh.Scene):
        bodies = []
        for node in obj.graph.nodes_geometry:
            transform, gname = obj.graph[node]
            mesh = obj.geometry[gname].copy()
            mesh.apply_transform(transform)
            bodies.append((str(node), np.asarray(mesh.triangles)))
        return bodies
    parts = obj.split(only_watertight=False)
    if len(parts) <= 1:
        return [(stem, np.asarray(obj.triangles))]
    return [(f"{stem} {k}", np.asarray(p.triangles)) for k, p in enumerate(parts, start=1)]


def write_ring_stack_step(path, rings):
    """Write rings [(name, r_in, r_out, z0, z1)] in metres as a STEP file of solids (mm),
    one named solid per ring. Used to make the example geometry."""
    import gmsh
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("rings")
        occ = gmsh.model.occ
        tags = []
        for name, r_in, r_out, z0, z1 in rings:
            mm = [v * 1e3 for v in (r_in, r_out, z0, z1)]
            outer = occ.addCylinder(0, 0, mm[2], 0, 0, mm[3] - mm[2], mm[1])
            inner = occ.addCylinder(0, 0, mm[2], 0, 0, mm[3] - mm[2], mm[0])
            (dim, tag), = occ.cut([(3, outer)], [(3, inner)])[0]
            tags.append((name, tag))
        occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()
    # gmsh does not write part names; put them into the STEP PRODUCT entries (one per
    # solid, in order) so CAD tools and load_bodies show "ring 1", "ring 2", ...
    import re
    text = open(path).read()
    names = iter([name for name, _ in tags])
    text = re.sub(r"PRODUCT\('Open CASCADE STEP translator [\d.]+ 1\.\d+',\s*'Open CASCADE STEP translator [\d.]+ 1\.\d+'",
                  lambda m: "PRODUCT('{0}','{0}'".format(next(names, "part")), text)
    open(path, "w").write(text)
