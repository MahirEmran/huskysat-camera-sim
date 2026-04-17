import os
import numpy as np
from Basilisk.utilities import macros, vizSupport


def add_box_triangles(vertices, faces, center, size):
    """Append a box mesh (12 triangles) to the vertex/face lists for OBJ export."""
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz

    box_vertices = [
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
    ]

    base = len(vertices) + 1
    vertices.extend(box_vertices)

    # Two triangles per face, with consistent winding.
    box_faces = [
        [base + 0, base + 1, base + 2], [base + 0, base + 2, base + 3],
        [base + 4, base + 6, base + 5], [base + 4, base + 7, base + 6],
        [base + 0, base + 4, base + 5], [base + 0, base + 5, base + 1],
        [base + 1, base + 5, base + 6], [base + 1, base + 6, base + 2],
        [base + 2, base + 6, base + 7], [base + 2, base + 7, base + 3],
        [base + 3, base + 7, base + 4], [base + 3, base + 4, base + 0],
    ]
    faces.extend(box_faces)


def add_cylinder_triangles(vertices, faces, center, radius, height, n_segs=16):
    """Append a capped cylinder (axis along Z) to the vertex/face lists."""
    cx, cy, cz = center
    base = len(vertices) + 1
    bot_z = cz - 0.25 * height
    top_z = cz + 0.5 * height

    # Build two rings: bottom then top.
    for z_val in (bot_z, top_z):
        for k in range(n_segs):
            ang = 2.0 * np.pi * k / n_segs
            vertices.append([cx + radius * np.cos(ang), cy + radius * np.sin(ang), z_val])

    vertices.append([cx, cy, bot_z])
    vertices.append([cx, cy, top_z])
    bot_cap = base + 2 * n_segs
    top_cap = base + 2 * n_segs + 1

    for k in range(n_segs):
        k1 = k
        k2 = (k + 1) % n_segs
        # Side wall (quad split into two triangles).
        faces.append([base + k1, base + k2, base + n_segs + k2])
        faces.append([base + k1, base + n_segs + k2, base + n_segs + k1])
        # Bottom and top caps.
        faces.append([bot_cap, base + k2, base + k1])
        faces.append([top_cap, base + n_segs + k1, base + n_segs + k2])


def add_panel_triangles(vertices, faces, center, size):
    """Append a panel as a thin box."""
    add_box_triangles(vertices, faces, center, size)


def write_obj(path, vertices, faces, face_materials, panel_open):
    """Write OBJ+MTL with component materials and double-sided triangles."""
    mtl_name = os.path.splitext(os.path.basename(path))[0] + ".mtl"
    mtl_path = os.path.join(os.path.dirname(path), mtl_name)

    with open(mtl_path, "w", encoding="utf-8") as mtl_file:
        mtl_file.write("newmtl body_mat\nKd 0.68 0.68 0.70\nKa 0.30 0.30 0.30\nKs 0.06 0.06 0.06\n\n")
        mtl_file.write("newmtl antenna_mat\nKd 0.50 0.50 0.52\nKa 0.22 0.22 0.22\nKs 0.03 0.03 0.03\n\n")
        if panel_open:
            mtl_file.write("newmtl panel_mat\nKd 0.10 0.44 0.92\nKa 0.08 0.10 0.20\nKs 0.03 0.04 0.08\n\n")
        else:
            mtl_file.write("newmtl panel_mat\nKd 0.56 0.56 0.58\nKa 0.24 0.24 0.24\nKs 0.03 0.03 0.03\n\n")

    # Store a face normal for each triangle to keep Vizard lighting stable.
    normals = []
    face_records = []
    for tri, mtl in zip(faces, face_materials):
        p0 = np.array(vertices[tri[0] - 1], dtype=float)
        p1 = np.array(vertices[tri[1] - 1], dtype=float)
        p2 = np.array(vertices[tri[2] - 1], dtype=float)
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n /= n_norm
        normals.append(n)
        face_records.append((tri, mtl, len(normals)))

    with open(path, "w", encoding="utf-8") as obj_file:
        obj_file.write(f"mtllib {mtl_name}\n")
        for v in vertices:
            obj_file.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in normals:
            obj_file.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        current_mtl = None
        for tri, mtl, n_idx in face_records:
            if mtl != current_mtl:
                obj_file.write(f"usemtl {mtl}\n")
                current_mtl = mtl
            # Emit both windings so the mesh remains visible from both sides.
            obj_file.write(f"f {tri[0]}//{n_idx} {tri[1]}//{n_idx} {tri[2]}//{n_idx}\n")
            obj_file.write(f"f {tri[0]}//{n_idx} {tri[2]}//{n_idx} {tri[1]}//{n_idx}\n")


def build_satellite_obj(path, panels_open, body_size_x_m, body_size_y_m, body_size_z_m):
    """Build full spacecraft mesh (body + antenna + panels) and write OBJ."""
    verts = []
    faces = []
    face_materials = []

    def mark_component(material_name, start_face_count):
        added = len(faces) - start_face_count
        if added > 0:
            face_materials.extend([material_name] * added)

    start_faces = len(faces)
    add_box_triangles(
        verts,
        faces,
        center=[0.0, 0.0, 0.0],
        size=[body_size_x_m, body_size_y_m, body_size_z_m],
    )
    mark_component("body_mat", start_faces)

    # Simple antenna stalk on the -Z side.
    ant_length = 0.08
    ant_radius = 0.005
    ant_cz = -0.5 * body_size_z_m - 0.5 * ant_length
    start_faces = len(faces)
    add_cylinder_triangles(
        verts,
        faces,
        center=[0.0, 0.0, ant_cz],
        radius=ant_radius,
        height=ant_length,
    )
    mark_component("antenna_mat", start_faces)

    # Panel geometry dimensions.
    panel_thickness = 0.003
    panel_short_edge = 0.60 * body_size_x_m
    panel_long_edge = body_size_z_m
    panel_z_elev = 0.0

    if panels_open:
        # Keep your current open-panel placement exactly as requested.
        panel_open_x = -0.5 * body_size_x_m - 0.5 * panel_short_edge
        panel_open_y = 0.5 * body_size_y_m + 0.5 * panel_thickness
        sign = -1.0
        start_faces = len(faces)
        add_panel_triangles(
            verts,
            faces,
            center=[panel_open_x, sign * panel_open_y, panel_z_elev],
            size=[panel_short_edge, panel_thickness, panel_long_edge],
        )
        add_panel_triangles(
            verts,
            faces,
            center=[panel_open_x + body_size_y_m + panel_short_edge, sign * panel_open_y, panel_z_elev],
            size=[panel_short_edge, panel_thickness, panel_long_edge],
        )
        mark_component("panel_mat", start_faces)

    write_obj(path, verts, faces, face_materials, panel_open=panels_open)


def apply_visual_model(viz, spacecraft_tag, model_path):
    """Replace spacecraft visual with a single custom model path."""
    # createCustomModel appends globally, so clear first to avoid model stacking.
    vizSupport.customModelList = []
    vizSupport.createCustomModel(
        viz,
        modelPath=model_path,
        rotation=[0.0, -90.0 * macros.D2R, 0.0],
        scale=[1.0, 1.0, 1.0],
        simBodiesToModify=[spacecraft_tag],
    )
    try:
        viz.settings.dataFresh = 1
    except Exception:
        pass
