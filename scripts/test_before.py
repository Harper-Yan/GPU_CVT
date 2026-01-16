import numpy as np
import trimesh
from scipy.spatial import cKDTree

def nr(x, e=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, e)

def tangent(n):
    n = nr(n)
    h = np.zeros_like(n)
    m = np.abs(n[:, 2]) > 0.9
    h[m] = [1, 0, 0]
    h[~m] = [0, 0, 1]
    u = nr(np.cross(h, n))
    v = nr(np.cross(n, u))
    return u, v

def knn_sites(sites, k=20):
    sites = np.ascontiguousarray(sites, dtype=np.float64)
    tree = cKDTree(sites)
    d, idx = tree.query(sites, k=k+1, workers=-1)
    return idx[:, 1:], d[:, 1:]

def clip_poly_halfspace(poly, a, b, c, eps=1e-12):
    if poly.shape[0] == 0:
        return poly
    def f(p): return a*p[0] + b*p[1] - c
    out = []
    s = poly[-1]
    fs = f(s)
    for e in poly:
        fe = f(e)
        ins_e = fe <= eps
        ins_s = fs <= eps
        if ins_e:
            if not ins_s:
                t = fs / (fs - fe + 1e-30)
                out.append(s + t*(e - s))
            out.append(e)
        elif ins_s:
            t = fs / (fs - fe + 1e-30)
            out.append(s + t*(e - s))
        s, fs = e, fe
    return np.asarray(out, dtype=np.float64)

def cell_poly2d(i, S, U, V, neigh, R):
    si = S[i]
    poly = np.array([[-R, -R], [ R, -R], [ R,  R], [-R,  R]], dtype=np.float64)
    ui = U[i]; vi = V[i]
    for j in neigh:
        d = S[j] - si
        a = float(d @ ui)
        b = float(d @ vi)
        c = 0.5 * float(d @ d)
        poly = clip_poly_halfspace(poly, a, b, c)
        if poly.shape[0] == 0:
            break
    return poly

def poly_area_centroid_2d(poly):
    if poly.shape[0] < 3:
        return 0.0, np.array([np.nan, np.nan], dtype=np.float64)
    x = poly[:, 0]; y = poly[:, 1]
    x2 = np.roll(x, -1); y2 = np.roll(y, -1)
    cr = x*y2 - x2*y
    A = 0.5 * np.sum(cr)
    if np.abs(A) < 1e-18:
        return 0.0, np.array([np.mean(x), np.mean(y)], dtype=np.float64)
    cx = (1.0/(6.0*A)) * np.sum((x + x2) * cr)
    cy = (1.0/(6.0*A)) * np.sum((y + y2) * cr)
    return A, np.array([cx, cy], dtype=np.float64)

def closest_point_tri(C, A, B, D):
    AB = B - A
    AC = D - A
    AP = C - A
    d1 = AB @ AP
    d2 = AC @ AP
    if d1 <= 0.0 and d2 <= 0.0: return A
    BP = C - B
    d3 = AB @ BP
    d4 = AC @ BP
    if d3 >= 0.0 and d4 <= d3: return B
    vc = d1*d4 - d3*d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3 + 1e-30)
        return A + v * AB
    CP = C - D
    d5 = AB @ CP
    d6 = AC @ CP
    if d6 >= 0.0 and d5 <= d6: return D
    vb = d5*d2 - d1*d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6 + 1e-30)
        return A + w * AC
    va = d3*d6 - d5*d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-30)
        return B + w * (D - B)
    denom = 1.0 / (va + vb + vc + 1e-30)
    v = vb * denom
    w = vc * denom
    return A + AB * v + AC * w

def update_to_mesh(C, tris_idx, P):
    if tris_idx.shape[0] == 0 or not np.isfinite(C).all():
        return C
    best = None
    best_d2 = np.inf
    for a, b, c in tris_idx:
        A = P[a].astype(np.float64); B = P[b].astype(np.float64); D = P[c].astype(np.float64)
        q = closest_point_tri(C, A, B, D)
        d = q - C
        d2 = float(d @ d)
        if d2 < best_d2:
            best_d2 = d2
            best = q
    return best if best is not None else C

def extract_faces_from_polys(S, N, U, V, idxn, polys2d, tol=1e-9):
    faces = set()
    for i in range(len(S)):
        poly = polys2d[i]
        if poly is None or poly.shape[0] < 3:
            continue
        ui = U[i]; vi = V[i]; si = S[i]
        neigh = idxn[i].astype(np.int64)
        D = S[neigh] - si
        A = D @ ui
        B = D @ vi
        C = 0.5 * np.einsum("ij,ij->i", D, D)
        for x, y in poly:
            s = A*x + B*y - C
            idx = np.argsort(np.abs(s))
            j1 = neigh[idx[0]]
            j2 = neigh[idx[1]]
            if j1 == j2:
                continue
            if np.abs(s[idx[0]]) > tol or np.abs(s[idx[1]]) > tol:
                continue
            a, b, c = int(i), int(j1), int(j2)
            if a == b or a == c or b == c:
                continue
            faces.add(tuple(sorted((a, b, c))))
    return np.array(list(faces), dtype=np.int64)

def orient_faces(F, Vpos, N):
    F = F.copy()
    for t in range(F.shape[0]):
        a, b, c = F[t]
        pa, pb, pc = Vpos[a], Vpos[b], Vpos[c]
        ntri = np.cross(pb - pa, pc - pa)
        ntgt = N[a] + N[b] + N[c]
        if (ntri @ ntgt) < 0.0:
            F[t] = [a, c, b]
    return F

def filter_degenerate(Vpos, F, eps=1e-12):
    if F.shape[0] == 0:
        return F
    A = Vpos[F[:,0]]; B = Vpos[F[:,1]]; C = Vpos[F[:,2]]
    e0 = np.linalg.norm(B-A, axis=1)
    e1 = np.linalg.norm(C-B, axis=1)
    e2 = np.linalg.norm(A-C, axis=1)
    area2 = np.linalg.norm(np.cross(B-A, C-A), axis=1)
    keep = (e0 > eps) & (e1 > eps) & (e2 > eps) & (area2 > eps)
    return F[keep]

def write_obj(path, Vpos, F):
    with open(path, "w") as f:
        for v in Vpos:
            f.write(f"v {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")
        for a, b, c in F:
            f.write(f"f {a+1} {b+1} {c+1}\n")

def lloyd_iter(S, P0, F0, N0, vf, treeP0, it, k_neigh=40, k_proj=10):
    d, idxN = treeP0.query(np.ascontiguousarray(S, dtype=np.float64), k=1, workers=-1)
    N = N0[idxN]
    U, V = tangent(N)
    idxn, _ = knn_sites(S, k=k_neigh)

    bbox = S.max(axis=0) - S.min(axis=0)
    R = 2.0 * 0.5 * float(np.max(bbox))

    polys2d = [None] * len(S)
    cent3d = np.full_like(S, np.nan, dtype=np.float64)
    for i in range(len(S)):
        poly = cell_poly2d(i, S, U, V, idxn[i], R)
        polys2d[i] = poly
        _, c2 = poly_area_centroid_2d(poly)
        if np.isfinite(c2).all():
            cent3d[i] = S[i] + c2[0]*U[i] + c2[1]*V[i]

    mask = np.isfinite(cent3d).all(axis=1)
    Snew = S.copy()
    if np.any(mask):
        dC, idxC = treeP0.query(np.ascontiguousarray(cent3d[mask], dtype=np.float64), k=k_proj, workers=-1)
        for t, i in enumerate(np.flatnonzero(mask)):
            nn = idxC[t]
            faces = set()
            for v in nn:
                for fi in vf[int(v)]:
                    faces.add(fi)
            faces = np.fromiter(faces, dtype=np.int64)
            tris = F0[faces] if faces.size else np.empty((0,3), dtype=np.int64)
            Snew[i] = update_to_mesh(cent3d[i], tris, P0)

    d2, idxN2 = treeP0.query(np.ascontiguousarray(Snew, dtype=np.float64), k=1, workers=-1)
    Nnew = N0[idxN2]
    Unew, Vnew = tangent(Nnew)
    idxn2, _ = knn_sites(Snew, k=k_neigh)
    polys2d2 = [None] * len(Snew)
    bbox2 = Snew.max(axis=0) - Snew.min(axis=0)
    R2 = 2.0 * 0.5 * float(np.max(bbox2))
    for i in range(len(Snew)):
        polys2d2[i] = cell_poly2d(i, Snew, Unew, Vnew, idxn2[i], R2)

    Fnew = extract_faces_from_polys(Snew, Nnew, Unew, Vnew, idxn2, polys2d2, tol=1e-9)
    Fnew = orient_faces(Fnew, Snew, Nnew)
    Fnew = filter_degenerate(Snew, Fnew, eps=1e-12)

    out_path = f"normal/iter_{it:03d}.obj"
    write_obj(out_path, Snew, Fnew)
    print("iter", it, "V", Snew.shape[0], "F", Fnew.shape[0], "wrote", out_path)
    return Snew

def main():
    in_path = "stanford-bunny.obj"
    iters = 15
    k_neigh = 32
    k_proj = 10

    m = trimesh.load(in_path, process=False)
    P0 = m.vertices.view(np.ndarray).astype(np.float64, copy=False)
    F0 = m.faces.view(np.ndarray).astype(np.int64, copy=False)
    N0 = m.vertex_normals.view(np.ndarray).astype(np.float64, copy=False)

    treeP0 = cKDTree(np.ascontiguousarray(P0, dtype=np.float64))
    vf = [[] for _ in range(len(P0))]
    for fi, (a, b, c) in enumerate(F0):
        vf[int(a)].append(fi); vf[int(b)].append(fi); vf[int(c)].append(fi)

    S = P0.copy()
    for it in range(iters):
        S = lloyd_iter(S, P0, F0, N0, vf, treeP0, it, k_neigh=k_neigh, k_proj=k_proj)

if __name__ == "__main__":
    main()