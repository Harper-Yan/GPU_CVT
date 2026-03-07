#!/usr/bin/env python3
"""
Visualize sites before/after projection and duplicate pairs.
Usage: python visualize_projection_debug.py <results_dir>
Example: python visualize_projection_debug.py results/armadillo/baseline
"""

import numpy as np
import sys
import os
from pathlib import Path

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False
    print("Warning: open3d not available, falling back to matplotlib")
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        print("Error: Neither open3d nor matplotlib available")
        sys.exit(1)


def load_obj_points(path):
    """Load points from OBJ file (only 'v' lines)."""
    points = []
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return np.array([])
    
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    points.append([x, y, z])
    return np.array(points, dtype=np.float64)


def load_duplicate_pairs(path):
    """Load duplicate pairs from text file."""
    pairs = []
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return pairs
    
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                idx1 = int(parts[0])
                idx2 = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                pairs.append({
                    'idx1': idx1,
                    'idx2': idx2,
                    'point': np.array([x, y, z], dtype=np.float64)
                })
    return pairs


def find_all_duplicate_groups(pairs, after):
    """Find all groups of duplicate points (connected components)."""
    if len(pairs) == 0:
        return []
    
    # Build adjacency graph of duplicate indices
    from collections import defaultdict
    graph = defaultdict(set)
    for pair in pairs:
        idx1, idx2 = pair['idx1'], pair['idx2']
        graph[idx1].add(idx2)
        graph[idx2].add(idx1)
    
    # Find all connected components (groups)
    if len(graph) == 0:
        return []
    
    visited = set()
    groups = []
    
    for start_idx in graph.keys():
        if start_idx in visited:
            continue
        
        # Find connected component starting from this index
        stack = [start_idx]
        group = set()
        
        while stack:
            idx = stack.pop()
            if idx in visited:
                continue
            visited.add(idx)
            group.add(idx)
            for neighbor in graph[idx]:
                if neighbor not in visited:
                    stack.append(neighbor)
        
        if len(group) > 0:
            groups.append(group)
    
    return groups


def find_local_surface_faces(group_points, mesh_vertices, mesh_faces, radius_factor=0.1):
    """Find faces that are near the duplicate group points (local surface)."""
    if len(group_points) == 0 or len(mesh_vertices) == 0 or len(mesh_faces) == 0:
        return []
    
    # Compute bounding box of group points
    group_min = group_points.min(axis=0)
    group_max = group_points.max(axis=0)
    group_center = (group_min + group_max) / 2.0
    group_size = np.max(group_max - group_min)
    
    # Use a radius based on the group size
    radius = group_size * radius_factor
    if radius < 1e-6:
        # Fallback: use a small fixed radius
        radius = np.max(np.linalg.norm(group_points - group_center, axis=1)) * 1.5
    
    # Find faces whose vertices or centroids are within radius of any group point
    local_faces = []
    
    for face_idx, face in enumerate(mesh_faces):
        if len(face) < 3:
            continue
        
        # Get triangle vertices
        v0 = mesh_vertices[face[0]]
        v1 = mesh_vertices[face[1]]
        v2 = mesh_vertices[face[2]]
        
        # Compute triangle centroid
        tri_center = (v0 + v1 + v2) / 3.0
        
        # Check if triangle is near any group point
        for group_point in group_points:
            dist = np.linalg.norm(tri_center - group_point)
            if dist <= radius:
                local_faces.append(face_idx)
                break
            
            # Also check if any vertex is near
            for v in [v0, v1, v2]:
                dist = np.linalg.norm(v - group_point)
                if dist <= radius:
                    local_faces.append(face_idx)
                    break
            if face_idx in local_faces:
                break
    
    return list(set(local_faces))  # Remove duplicates


def load_obj_mesh(path):
    """Load mesh (vertices and faces) from OBJ file."""
    vertices = []
    faces = []
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return np.array([]), np.array([])
    
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
            elif line.startswith('f '):
                parts = line.strip().split()
                if len(parts) >= 4:
                    # OBJ faces are 1-indexed, convert to 0-indexed
                    face_indices = []
                    for part in parts[1:]:
                        # Handle format like "1" or "1/1/1"
                        idx = int(part.split('/')[0]) - 1
                        face_indices.append(idx)
                    if len(face_indices) >= 3:
                        # Triangulate if needed (simple: use first vertex with all others)
                        for i in range(1, len(face_indices) - 1):
                            faces.append([face_indices[0], face_indices[i], face_indices[i+1]])
    
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def visualize_with_open3d(before, after, dup_pairs, mesh_vertices, mesh_faces, all_groups):
    """Visualize using Open3D - iteratively show all duplicate groups."""
    if len(all_groups) == 0:
        print("No duplicate groups found")
        return
    
    print(f"Found {len(all_groups)} duplicate groups. Showing each group iteratively...")
    print("Press 'N' for next group, 'P' for previous group, 'Q' to quit")
    
    current_group_idx = 0
    
    def key_callback(vis, key, action):
        nonlocal current_group_idx
        if action == 0:  # Key press
            if key == ord('N') or key == ord('n'):
                current_group_idx = (current_group_idx + 1) % len(all_groups)
                update_visualization(vis, current_group_idx)
                return True
            elif key == ord('P') or key == ord('p'):
                current_group_idx = (current_group_idx - 1) % len(all_groups)
                update_visualization(vis, current_group_idx)
                return True
            elif key == ord('Q') or key == ord('q'):
                vis.close()
                return True
        return False
    
    def update_visualization(vis, group_idx):
        # Clear all geometries
        vis.clear_geometries()
        
        # Show current group
        current_group = all_groups[group_idx]
        group_points = np.array([after[i] for i in current_group if i < len(after)]) if len(current_group) > 0 and len(after) > 0 else np.array([])
        
        # Find local surface faces around the duplicate group
        local_face_indices = []
        if len(group_points) > 0 and len(mesh_vertices) > 0 and len(mesh_faces) > 0:
            local_face_indices = find_local_surface_faces(group_points, mesh_vertices, mesh_faces)
        
        # Re-add full mesh (dimmed)
        if len(mesh_vertices) > 0 and len(mesh_faces) > 0:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
            mesh.triangles = o3d.utility.Vector3iVector(mesh_faces)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Dark gray for background
            vis.add_geometry(mesh, reset_bounding_box=False)
        
        # Highlight local surface faces
        if len(local_face_indices) > 0 and len(mesh_vertices) > 0 and len(mesh_faces) > 0:
            local_faces = mesh_faces[local_face_indices]
            local_mesh = o3d.geometry.TriangleMesh()
            local_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
            local_mesh.triangles = o3d.utility.Vector3iVector(local_faces)
            local_mesh.compute_vertex_normals()
            colors = [
                [1.0, 0.3, 0.3],  # Light red
                [0.3, 1.0, 0.3],  # Light green
                [0.3, 0.3, 1.0],  # Light blue
                [1.0, 1.0, 0.3],  # Light yellow
                [1.0, 0.3, 1.0],  # Light magenta
                [0.3, 1.0, 1.0],  # Light cyan
            ]
            color = colors[group_idx % len(colors)]
            local_mesh.paint_uniform_color(color)
            vis.add_geometry(local_mesh, reset_bounding_box=False)
        
        # Re-add base points (all projected points, dimmed)
        if len(after) > 0:
            pc_after = o3d.geometry.PointCloud()
            pc_after.points = o3d.utility.Vector3dVector(after)
            pc_after.paint_uniform_color([0.2, 0.2, 0.2])  # Very dark gray
            vis.add_geometry(pc_after, reset_bounding_box=False)
        
        # Show current group points
        if len(group_points) > 0:
            # Highlight current duplicate group points
            pc_dup = o3d.geometry.PointCloud()
            pc_dup.points = o3d.utility.Vector3dVector(group_points)
            # Use different colors for different groups
            colors = [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
                [0.0, 1.0, 1.0],  # Cyan
            ]
            color = colors[group_idx % len(colors)]
            pc_dup.paint_uniform_color(color)
            vis.add_geometry(pc_dup, reset_bounding_box=False)
            
            # Create lines connecting points in current group
            line_points = []
            line_indices = []
            
            for pair in dup_pairs:
                idx1, idx2 = pair['idx1'], pair['idx2']
                if idx1 in current_group and idx2 in current_group:
                    if idx1 < len(after) and idx2 < len(after):
                        p1 = after[idx1]
                        p2 = after[idx2]
                        start_idx = len(line_points)
                        line_points.append(p1)
                        line_points.append(p2)
                        line_indices.append([start_idx, start_idx + 1])
            
            if len(line_points) > 0:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
                line_set.lines = o3d.utility.Vector2iVector(np.array(line_indices))
                line_set.paint_uniform_color(color)
                vis.add_geometry(line_set, reset_bounding_box=False)
        
        # Update window title
        vis.poll_events()
        vis.update_renderer()
        print(f"Showing group {group_idx + 1}/{len(all_groups)} ({len(current_group)} points, {len(local_face_indices)} local faces)")
    
    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Duplicate Groups (1/{len(all_groups)})", width=1920, height=1080)
    
    # Register key callback
    vis.register_key_callback(ord('N'), key_callback)
    vis.register_key_callback(ord('n'), key_callback)
    vis.register_key_callback(ord('P'), key_callback)
    vis.register_key_callback(ord('p'), key_callback)
    vis.register_key_callback(ord('Q'), key_callback)
    vis.register_key_callback(ord('q'), key_callback)
    
    # Initial visualization
    update_visualization(vis, 0)
    
    opt = vis.get_render_option()
    opt.point_size = 5.0
    opt.line_width = 3.0
    
    vis.run()
    vis.destroy_window()


def visualize_with_matplotlib(before, after, dup_pairs, mesh_vertices, mesh_faces, all_groups):
    """Visualize using matplotlib - show all groups with local surfaces."""
    if len(all_groups) == 0:
        print("No duplicate groups found")
        return
    
    # Create figure with subplots
    n_groups_to_show = min(len(all_groups), 6)  # Show up to 6 groups
    n_cols = 3
    n_rows = (n_groups_to_show + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(18, 6 * n_rows))
    
    colors = [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
    ]
    
    light_colors = [
        [1.0, 0.5, 0.5],  # Light red
        [0.5, 1.0, 0.5],  # Light green
        [0.5, 0.5, 1.0],  # Light blue
        [1.0, 1.0, 0.5],  # Light yellow
        [1.0, 0.5, 1.0],  # Light magenta
        [0.5, 1.0, 1.0],  # Light cyan
    ]
    
    for group_idx, group in enumerate(all_groups[:n_groups_to_show]):
        ax = fig.add_subplot(n_rows, n_cols, group_idx + 1, projection='3d')
        
        group_points = np.array([after[i] for i in group if i < len(after)]) if len(group) > 0 and len(after) > 0 else np.array([])
        
        # Find local surface faces
        local_face_indices = []
        if len(group_points) > 0 and len(mesh_vertices) > 0 and len(mesh_faces) > 0:
            local_face_indices = find_local_surface_faces(group_points, mesh_vertices, mesh_faces)
        
        # Show full mesh wireframe (dimmed)
        if len(mesh_vertices) > 0 and len(mesh_faces) > 0:
            for face_idx, face in enumerate(mesh_faces[:1000]):  # Limit for performance
                if len(face) >= 3:
                    triangle = mesh_vertices[face]
                    triangle_closed = np.vstack([triangle, triangle[0]])
                    ax.plot(triangle_closed[:, 0], triangle_closed[:, 1], triangle_closed[:, 2], 
                           'lightgray', linewidth=0.2, alpha=0.1)
        
        # Highlight local surface faces
        if len(local_face_indices) > 0 and len(mesh_vertices) > 0 and len(mesh_faces) > 0:
            light_color = light_colors[group_idx % len(light_colors)]
            for face_idx in local_face_indices[:200]:  # Limit for performance
                if face_idx < len(mesh_faces):
                    face = mesh_faces[face_idx]
                    if len(face) >= 3:
                        triangle = mesh_vertices[face]
                        triangle_closed = np.vstack([triangle, triangle[0]])
                        ax.plot(triangle_closed[:, 0], triangle_closed[:, 1], triangle_closed[:, 2], 
                               color=light_color, linewidth=1.5, alpha=0.6)
        
        # Show all projected points (dimmed)
        if len(after) > 0:
            ax.scatter(after[:, 0], after[:, 1], after[:, 2], 
                      c='lightblue', s=0.5, alpha=0.1, label='All points')
        
        # Show current duplicate group
        if len(group_points) > 0:
            color = colors[group_idx % len(colors)]
            ax.scatter(group_points[:, 0], group_points[:, 1], group_points[:, 2], 
                      c=[color], s=100, alpha=1.0, 
                      label=f'Group {group_idx + 1} ({len(group)} points)')
            
            # Draw lines connecting points in current group
            for pair in dup_pairs:
                idx1, idx2 = pair['idx1'], pair['idx2']
                if idx1 in group and idx2 in group:
                    if idx1 < len(after) and idx2 < len(after):
                        p1 = after[idx1]
                        p2 = after[idx2]
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                               color=color, linewidth=2.0, alpha=0.8)
        
        ax.set_title(f'Group {group_idx + 1}/{len(all_groups)}: {len(group)} pts, {len(local_face_indices)} local faces')
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    if len(all_groups) > n_groups_to_show:
        print(f"Note: Showing first {n_groups_to_show} of {len(all_groups)} groups")
    
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_projection_debug.py <results_dir> [original_mesh.obj]")
        print("Example: python visualize_projection_debug.py results/armadillo/baseline objs/armadillo.obj")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    original_mesh_path = sys.argv[2] if len(sys.argv) >= 3 else None
    
    # Load data
    before_path = os.path.join(results_dir, "debug_before_projection.obj")
    after_path = os.path.join(results_dir, "debug_after_projection.obj")
    pairs_path = os.path.join(results_dir, "debug_duplicate_pairs.txt")
    
    print(f"Loading data from {results_dir}...")
    before = load_obj_points(before_path)
    after = load_obj_points(after_path)
    dup_pairs = load_duplicate_pairs(pairs_path)
    
    # Load original mesh if provided
    mesh_vertices = np.array([])
    mesh_faces = np.array([])
    if original_mesh_path and os.path.exists(original_mesh_path):
        print(f"Loading original mesh from {original_mesh_path}...")
        mesh_vertices, mesh_faces = load_obj_mesh(original_mesh_path)
        print(f"Loaded mesh: {len(mesh_vertices)} vertices, {len(mesh_faces)} faces")
    else:
        print("Warning: Original mesh not provided or not found. Mesh visualization will be skipped.")
    
    print(f"Loaded {len(before)} points before projection")
    print(f"Loaded {len(after)} points after projection")
    print(f"Loaded {len(dup_pairs)} duplicate pairs")
    
    if len(before) == 0 and len(after) == 0:
        print("Error: No data found. Make sure debug files exist.")
        sys.exit(1)
    
    # Find all duplicate groups
    all_groups = find_all_duplicate_groups(dup_pairs, after)
    print(f"Found {len(all_groups)} duplicate groups")
    for i, group in enumerate(all_groups):
        print(f"  Group {i + 1}: {len(group)} points")
    
    if len(all_groups) == 0:
        print("No duplicate groups found")
        sys.exit(0)
    
    # Visualize
    if HAS_O3D:
        print("Visualizing with Open3D...")
        print("Controls: N/Next - next group, P/Previous - previous group, Q/Quit - exit")
        visualize_with_open3d(before, after, dup_pairs, mesh_vertices, mesh_faces, all_groups)
    elif HAS_MATPLOTLIB:
        print("Visualizing with matplotlib...")
        visualize_with_matplotlib(before, after, dup_pairs, mesh_vertices, mesh_faces, all_groups)
    else:
        print("Error: No visualization library available")
        sys.exit(1)


if __name__ == "__main__":
    main()
