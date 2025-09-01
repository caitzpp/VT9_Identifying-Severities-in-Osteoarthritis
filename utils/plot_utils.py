import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def plotly_hdbscan_highlight_kl(
    X,                    # (n,2 or n,3) embeddings
    labels,               # cluster labels (int; -1 = noise)
    y_kl,                 # KL-score per point (int)
    probabilities=None,   # optional [0..1] for size scaling
    dim=2,                # 2 or 3
    title="UMAP + HDBSCAN (KL highlight)",
    base_gray_opacity=0.2,
    base_gray_size=6,
    size_min=8,
    size_max=20,
):
    X = np.asarray(X)
    labels = np.asarray(labels)
    y_kl = np.asarray(y_kl)
    n = X.shape[0]
    assert dim in (2, 3), "dim must be 2 or 3"
    assert X.shape[1] >= dim, "X must have at least `dim` columns"

    # Sizes from probabilities (optional)
    if probabilities is None:
        sizes = np.full(n, (size_min + size_max) / 2.0, dtype=float)
    else:
        p = np.asarray(probabilities, dtype=float)
        pmin, pmax = p.min(), p.max()
        if pmax > 1.0 or pmin < 0.0:
            p = (p - pmin) / (pmax - pmin + 1e-12)
        p = np.nan_to_num(p, nan=0.7, posinf=1.0, neginf=0.0)
        sizes = size_min + (size_max - size_min) * p

    # Stable cluster colors
    unique_clusters = sorted(set(labels) - {-1})
    cmap = px.colors.qualitative.Safe  # nice, readable palette
    # expand palette if needed
    while len(cmap) < max(1, len(unique_clusters)):
        cmap = cmap + cmap
    cluster_color = {c: cmap[i % len(cmap)] for i, c in enumerate(unique_clusters)}
    cluster_color[-1] = "black"  # noise

    # Base gray layer (all points)
    traces = []
    x, y = X[:, 0], X[:, 1]
    if dim == 2:
        base = go.Scattergl(
            x=x, y=y,
            mode="markers",
            marker=dict(color="lightgray", size=base_gray_size),
            opacity=base_gray_opacity,
            name="All (gray)",
            showlegend=False
        )
    else:
        base = go.Scatter3d(
            x=x, y=y, z=X[:, 2],
            mode="markers",
            marker=dict(color="lightgray", size=max(2, int(base_gray_size/2))),
            opacity=base_gray_opacity,
            name="All (gray)",
            showlegend=False
        )
    traces.append(base)

    # Colored layers: one trace per (KL, cluster)
    kl_values = sorted(np.unique(y_kl))
    # We’ll show cluster legend only for the first KL so we don’t duplicate legend entries.
    show_cluster_legend_for_kl = kl_values[0] if len(kl_values) else None

    # Keep track of visibility arrays for dropdown
    # Index 0 is the base gray layer (always visible)
    # For each KL we’ll build a visibility mask
    vis_masks = []
    # placeholder for all colored traces indices (to set visibility later)
    colored_trace_indices_by_kl = {kl: [] for kl in kl_values}

    for kl in kl_values:
        for c in sorted(set(labels)):
            mask = (y_kl == kl) & (labels == c)
            if not np.any(mask):
                continue

            name = f"Cluster {c}" if c != -1 else "Noise"
            showlegend = bool(kl == show_cluster_legend_for_kl)  # legend once
            color = cluster_color.get(c, "gray")

            if dim == 2:
                tr = go.Scattergl(
                    x=x[mask], y=y[mask],
                    mode="markers",
                    marker=dict(
                        color=color,
                        size=sizes[mask],
                        line=dict(width=0.5, color="black")
                    ),
                    name=name,
                    showlegend=showlegend,
                )
            else:
                tr = go.Scatter3d(
                    x=x[mask], y=y[mask], z=X[mask, 2],
                    mode="markers",
                    marker=dict(
                        color=color,
                        size=np.clip(sizes[mask]/3.0, 2, 8)  # keep 3D sizes reasonable
                    ),
                    name=name,
                    showlegend=showlegend,
                )

            traces.append(tr)
            colored_trace_indices_by_kl[kl].append(len(traces) - 1)

    # Build visibility masks for dropdown
    total_traces = len(traces)
    # Helper to build vis array for a given kl
    def vis_for_kl(kl):
        vis = [True] + [False] * (total_traces - 1)  # base gray visible
        for idx in colored_trace_indices_by_kl.get(kl, []):
            vis[idx] = True
        return vis

    # Buttons: one per KL + "All off" + "All on"
    buttons = []
    for kl in kl_values:
        buttons.append(dict(
            label=f"KL = {kl}",
            method="update",
            args=[
                {"visible": vis_for_kl(kl)},
                {"title": f"{title} — Highlight KL={kl}"}
            ]
        ))

    # Optional utility buttons
    # Show none (only gray)
    buttons.append(dict(
        label="None",
        method="update",
        args=[
            {"visible": [True] + [False]*(total_traces-1)},
            {"title": f"{title} — Highlight: None"}
        ]
    ))
    # Show all colored (every KL)
    buttons.append(dict(
        label="All",
        method="update",
        args=[
            {"visible": [True] + [True]*(total_traces-1)},
            {"title": f"{title} — Highlight: All KL"}
        ]
    ))

    # Initial visibility: just gray (or pick the first KL if you prefer)
    initial_visible = [True] + [False]*(total_traces-1)
    for i, tr in enumerate(traces):
        tr.visible = initial_visible[i]

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=f"{title} — Highlight: None",
        legend=dict(
            title="Clusters",
            itemclick="toggle",       # toggle individual cluster
            itemdoubleclick="toggleothers"
        ),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            showactive=True,
            x=1.02, y=1.0, xanchor="left", yanchor="top",
            buttons=buttons
        )],
        margin=dict(l=40, r=120, t=60, b=40),
    )

    if dim == 3:
        fig.update_layout(
            scene=dict(
                xaxis_title="dim 0",
                yaxis_title="dim 1",
                zaxis_title="dim 2",
                camera=dict(eye=dict(x=1.6, y=1.6, z=0.9))
            )
        )
    else:
        fig.update_xaxes(title="dim 0")
        fig.update_yaxes(title="dim 1")

    return fig


def plotly_hdbscan_highlight_kl2(
    X,
    labels,
    y_kl,
    probabilities=None,
    dim=2,
    title="UMAP + HDBSCAN (KL highlight)",
    base_gray_opacity=0.2,
    base_gray_size=6,
    size_min=8,
    size_max=20,
    # --- compact controls ---
    compact=True,          # tighten axes & margins
    q_clip=0.01,           # clip 1% tails on each side (per axis)
    pad_frac=0.03,         # small padding around data
    equal_aspect=True,     # keep equal aspect in 2D
    zoom_by_kl=False,      # when selecting a KL, update axes to that subset
):
    X = np.asarray(X)
    labels = np.asarray(labels)
    y_kl = np.asarray(y_kl)
    n = X.shape[0]
    assert dim in (2, 3), "dim must be 2 or 3"
    assert X.shape[1] >= dim, "X must have at least `dim` columns"

    # Sizes from probabilities (with NaN/inf safety)
    if probabilities is None:
        sizes = np.full(n, (size_min + size_max) / 2.0, dtype=float)
    else:
        p = np.asarray(probabilities, dtype=float)
        p = np.nan_to_num(p, nan=0.7, posinf=1.0, neginf=0.0)
        pmin, pmax = p.min(), p.max()
        if pmax > 1.0 or pmin < 0.0:
            rng = pmax - pmin
            p = np.zeros_like(p) if rng == 0 else (p - pmin) / (rng + 1e-12)
        sizes = size_min + (size_max - size_min) * p
    sizes = np.nan_to_num(sizes, nan=size_min, posinf=size_max, neginf=size_min)

    # Stable cluster colors
    unique_clusters = sorted(set(labels) - {-1})
    cmap = px.colors.qualitative.Safe
    while len(cmap) < max(1, len(unique_clusters)):
        cmap = cmap + cmap
    cluster_color = {c: cmap[i % len(cmap)] for i, c in enumerate(unique_clusters)}
    cluster_color[-1] = "black"

    # Base gray layer
    traces = []
    x, y = X[:, 0], X[:, 1]
    if dim == 2:
        base = go.Scattergl(
            x=x, y=y, mode="markers",
            marker=dict(color="lightgray", size=base_gray_size),
            opacity=base_gray_opacity, name="All (gray)", showlegend=False
        )
    else:
        base = go.Scatter3d(
            x=x, y=y, z=X[:, 2], mode="markers",
            marker=dict(color="lightgray", 
                        #size=max(2, int(base_gray_size/2))
            ),
            opacity=base_gray_opacity, name="All (gray)", showlegend=False
        )
    traces.append(base)

    # Colored layers per (KL, cluster)
    kl_values = sorted(np.unique(y_kl))
    show_cluster_legend_for_kl = kl_values[0] if len(kl_values) else None
    colored_trace_indices_by_kl = {kl: [] for kl in kl_values}

    for kl in kl_values:
        for c in sorted(set(labels)):
            mask = (y_kl == kl) & (labels == c)
            if not np.any(mask):
                continue
            name = f"Cluster {c}" if c != -1 else "Noise"
            showlegend = bool(kl == show_cluster_legend_for_kl)
            color = cluster_color.get(c, "gray")

            if dim == 2:
                tr = go.Scattergl(
                    x=x[mask], y=y[mask], mode="markers",
                    marker=dict(
                        color=color,
                        size=(sizes[mask] if c != -1 else size_min),
                        line=dict(width=0.5, color="black") if c != -1 else None
                    ),
                    name=name, showlegend=showlegend,
                )
            else:
                tr = go.Scatter3d(
                    x=x[mask], y=y[mask], z=X[mask, 2], mode="markers",
                    marker=dict(
                        color=color,
                        #size=(np.clip(sizes[mask]/3.0, 2, 8) if c != -1 else max(2, int(size_min/2)))
                    ),
                    name=name, showlegend=showlegend,
                )
            traces.append(tr)
            colored_trace_indices_by_kl[kl].append(len(traces) - 1)

    # Helper: compact axis ranges (percentile clip + padding)
    def _range(vec):
        lo, hi = np.nanpercentile(vec, [100*q_clip, 100*(1 - q_clip)])
        pad = (hi - lo) * pad_frac
        if pad == 0:
            pad = max(1e-6, abs(hi) * pad_frac + abs(lo) * pad_frac)
        return [float(lo - pad), float(hi + pad)]

    # Global compact ranges (used initially and when zoom_by_kl=False)
    if compact:
        rx_all = _range(X[:, 0])
        ry_all = _range(X[:, 1])
        if dim == 3:
            rz_all = _range(X[:, 2])

    # KL-specific ranges (optional)
    ranges_by_kl = {}
    if compact and zoom_by_kl:
        for kl in kl_values:
            m = (y_kl == kl)
            if np.any(m):
                rxd = _range(X[m, 0])
                ryd = _range(X[m, 1])
                if dim == 3:
                    rzd = _range(X[m, 2])
                    ranges_by_kl[kl] = dict(x=rxd, y=ryd, z=rzd)
                else:
                    ranges_by_kl[kl] = dict(x=rxd, y=ryd)

    # Visibility masks
    total_traces = len(traces)
    def vis_for_kl(kl):
        vis = [True] + [False]*(total_traces - 1)
        for idx in colored_trace_indices_by_kl.get(kl, []):
            vis[idx] = True
        return vis

    # Buttons
    buttons = []
    for kl in kl_values:
        layout_updates = {"title": f"{title} — Highlight KL={kl}"}
        if compact and zoom_by_kl:
            if dim == 3:
                r = ranges_by_kl.get(kl, dict(x=rx_all, y=ry_all, z=rz_all))
                layout_updates.update(scene=dict(
                    xaxis=dict(range=r["x"]), yaxis=dict(range=r["y"]), zaxis=dict(range=r["z"]),
                    aspectmode="cube"
                ))
            else:
                r = ranges_by_kl.get(kl, dict(x=rx_all, y=ry_all))
                layout_updates.update(xaxis=dict(range=r["x"]),
                                      yaxis=dict(range=r["y"]))
        buttons.append(dict(
            label=f"KL = {kl}",
            method="update",
            args=[{"visible": vis_for_kl(kl)}, layout_updates]
        ))

    buttons.append(dict(
        label="None",
        method="update",
        args=[{"visible": [True] + [False]*(total_traces-1)},
              {"title": f"{title} — Highlight: None"}]
    ))
    buttons.append(dict(
        label="All",
        method="update",
        args=[{"visible": [True] + [True]*(total_traces-1)},
              {"title": f"{title} — Highlight: All KL"}]
    ))

    # Initial visibility
    initial_visible = [True] + [False]*(total_traces-1)
    for i, tr in enumerate(traces):
        tr.visible = initial_visible[i]

    fig = go.Figure(data=traces)

    # Layout + compact tweaks
    layout_kwargs = dict(
        title=f"{title} — Highlight: None",
        legend=dict(title="Clusters", itemclick="toggle", itemdoubleclick="toggleothers"),
        updatemenus=[dict(
            type="dropdown", direction="down", showactive=True,
            x=0.02, y=1.12, xanchor="left", yanchor="top",  # tuck inside to save right margin
            buttons=buttons
        )],
        margin=dict(l=30, r=30, t=80, b=30),  # tighter margins,
    )
    fig.update_layout(**layout_kwargs)

    if dim == 3:
        scene_dict = dict(
            xaxis_title="dim 0", yaxis_title="dim 1", zaxis_title="dim 2",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        )
        if compact:
            # compact ranges
            scene_dict.update(
                xaxis=dict(range=rx_all),
                yaxis=dict(range=ry_all),
                zaxis=dict(range=rz_all),
                aspectmode="cube"  # equal axes -> compact cube
            )
        fig.update_layout(scene=scene_dict)
    else:
        fig.update_xaxes(title="dim 0")
        fig.update_yaxes(title="dim 1")
        if compact:
            fig.update_xaxes(range=rx_all, constrain="domain")
            fig.update_yaxes(range=ry_all, constrain="domain")
            if equal_aspect:
                fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig

def plotly_hdbscan_highlight(
    X,
    labels,
    y_kl,
    probabilities=None,
    dim=2,
    title="UMAP + HDBSCAN (highlight)",
    # legend / buttons / colors
    legend_by="cluster",     # "cluster" or "kl"
    buttons_by="kl",         # "cluster" or "kl"
    color_by="legend",       # "legend" or explicitly "cluster"/"kl"
    # gray base & sizes
    base_gray_opacity=0.2,
    base_gray_size=6,
    size_min=8,
    size_max=20,
    # compact controls
    compact=True,
    q_clip=0.01,
    pad_frac=0.03,
    equal_aspect=True,
    zoom_by=None,            # None, "kl", or "cluster"
):
    """
    Interactive scatter (2D/3D) with:
      - A gray base layer (all points)
      - Colored traces split by (KL, Cluster)
      - A dropdown that toggles visibility by `buttons_by`
      - A legend grouping by `legend_by`
      - Colors chosen by `color_by` (default ties to legend)

    labels: cluster labels (int; -1 = noise)
    y_kl:   KL-scores (int)
    """

    # ---------- inputs & checks ----------
    X = np.asarray(X)
    labels = np.asarray(labels)
    y_kl = np.asarray(y_kl)
    n = X.shape[0]
    assert dim in (2, 3), "dim must be 2 or 3"
    assert X.shape[1] >= dim, "X must have at least `dim` columns"
    if labels.shape[0] != n or y_kl.shape[0] != n:
        raise ValueError("X, labels, and y_kl must have same length")

    # ---------- sizes from probabilities (robust) ----------
    if probabilities is None:
        sizes = np.full(n, (size_min + size_max) / 2.0, dtype=float)
    else:
        p = np.asarray(probabilities, dtype=float)
        if p.shape[0] != n:
            raise ValueError("'probabilities' length mismatch")
        p = np.nan_to_num(p, nan=0.7, posinf=1.0, neginf=0.0)
        pmin, pmax = float(np.min(p)), float(np.max(p))
        if pmax > 1.0 or pmin < 0.0:
            rng = pmax - pmin
            p = np.zeros_like(p) if rng == 0 else (p - pmin) / (rng + 1e-12)
        sizes = size_min + (size_max - size_min) * p
    sizes = np.nan_to_num(sizes, nan=size_min, posinf=size_max, neginf=size_min)

    # ---------- categories ----------
    clusters = sorted(set(labels))
    kl_values = sorted(set(y_kl))

    # ---------- color maps ----------
    # choose which category drives colors
    if color_by == "legend":
        color_by = legend_by
    if color_by not in ("cluster", "kl"):
        raise ValueError("color_by must be 'legend', 'cluster', or 'kl'")

    # palettes
    pal = px.colors.qualitative.Safe
    while len(pal) < max(1, max(len(set(labels)-{-1}), len(kl_values))):
        pal = pal + pal

    cluster_color = {}
    if len(set(labels) - {-1}) == 0:
        cluster_color = {}
    else:
        for i, c in enumerate(sorted(set(labels) - {-1})):
            cluster_color[c] = pal[i % len(pal)]
    cluster_color[-1] = "black"  # noise stays black

    kl_color = {kl: pal[i % len(pal)] for i, kl in enumerate(kl_values)}

    def color_for(c, kl):
        if c == -1:
            return "black"
        if color_by == "cluster":
            return cluster_color.get(c, "gray")
        else:
            return kl_color.get(kl, "gray")

    # ---------- base gray layer ----------
    traces = []
    x, y = X[:, 0], X[:, 1]
    if dim == 2:
        base = go.Scattergl(
            x=x, y=y, mode="markers",
            marker=dict(color="lightgray", size=base_gray_size),
            opacity=base_gray_opacity, name="All (gray)", showlegend=False
        )
    else:
        base = go.Scatter3d(
            x=x, y=y, z=X[:, 2], mode="markers",
            marker=dict(color="lightgray", 
                        #size=max(2, int(base_gray_size/2))
                        ),
            opacity=base_gray_opacity, name="All (gray)", showlegend=False
        )
    traces.append(base)

    # ---------- colored traces: one per (KL, cluster) ----------
    # we’ll attach legend to one dimension and dropdown to the other
    # showlegend only for the first occurrence of each legend item
    seen_legend_items = set()

    # Keep indices for dropdown control
    indices_by_kl = {kl: [] for kl in kl_values}
    indices_by_cluster = {c: [] for c in clusters}

    for kl in kl_values:
        for c in clusters:
            mask = (y_kl == kl) & (labels == c)
            if not np.any(mask):
                continue

            # legend naming/grouping
            if legend_by == "cluster":
                legend_name = f"Cluster {c}" if c != -1 else "Noise"
                legend_group = f"cluster:{c}"
            else:
                legend_name = f"KL = {kl}"
                legend_group = f"kl:{kl}"

            showlegend = False
            if legend_name not in seen_legend_items:
                showlegend = True
                seen_legend_items.add(legend_name)

            col = color_for(c, kl)

            if dim == 2:
                tr = go.Scattergl(
                    x=x[mask], y=y[mask], mode="markers",
                    marker=dict(
                        color=col,
                       # size=(sizes[mask] if c != -1 else size_min),
                        line=dict(width=0.5, color="black") if c != -1 else None
                    ),
                    name=legend_name,
                    legendgroup=legend_group,
                    showlegend=bool(showlegend),
                )
            else:
                tr = go.Scatter3d(
                    x=x[mask], y=y[mask], z=X[mask, 2], mode="markers",
                    marker=dict(
                        color=col,
                       # size=(np.clip(sizes[mask]/3.0, 2, 8) if c != -1 else max(2, int(size_min/2)))
                    ),
                    name=legend_name,
                    legendgroup=legend_group,
                    showlegend=bool(showlegend),
                )

            traces.append(tr)
            idx = len(traces) - 1
            indices_by_kl[kl].append(idx)
            indices_by_cluster[c].append(idx)

    # ---------- compact axis ranges ----------
    def _range(vec):
        lo, hi = np.nanpercentile(vec, [100*q_clip, 100*(1 - q_clip)])
        pad = (hi - lo) * pad_frac
        if pad == 0:
            pad = max(1e-6, abs(hi) * pad_frac + abs(lo) * pad_frac)
        return [float(lo - pad), float(hi + pad)]

    if compact:
        rx_all = _range(X[:, 0])
        ry_all = _range(X[:, 1])
        if dim == 3:
            rz_all = _range(X[:, 2])

    # per-category ranges for zoom_by
    ranges_by_kl, ranges_by_cluster = {}, {}
    if compact and zoom_by in ("kl", "cluster"):
        if zoom_by == "kl":
            for kl in kl_values:
                m = (y_kl == kl)
                if np.any(m):
                    rx, ry = _range(X[m, 0]), _range(X[m, 1])
                    if dim == 3:
                        rz = _range(X[m, 2])
                        ranges_by_kl[kl] = dict(x=rx, y=ry, z=rz)
                    else:
                        ranges_by_kl[kl] = dict(x=rx, y=ry)
        else:
            for c in clusters:
                m = (labels == c)
                if np.any(m):
                    rx, ry = _range(X[m, 0]), _range(X[m, 1])
                    if dim == 3:
                        rz = _range(X[m, 2])
                        ranges_by_cluster[c] = dict(x=rx, y=ry, z=rz)
                    else:
                        ranges_by_cluster[c] = dict(x=rx, y=ry)

    # ---------- visibility helpers ----------
    total_traces = len(traces)

    def vis_for_kl(kl):
        vis = [True] + [False]*(total_traces - 1)
        for idx in indices_by_kl.get(kl, []):
            vis[idx] = True
        return vis

    def vis_for_cluster(c):
        vis = [True] + [False]*(total_traces - 1)
        for idx in indices_by_cluster.get(c, []):
            vis[idx] = True
        return vis

    # ---------- dropdown buttons ----------
    buttons = []
    if buttons_by == "kl":
        for kl in kl_values:
            layout_updates = {"title": f"{title} — Highlight KL={kl}"}
            if compact and zoom_by == "kl":
                if dim == 3:
                    r = ranges_by_kl.get(kl, dict(x=rx_all, y=ry_all, z=rz_all))
                    layout_updates.update(scene=dict(
                        xaxis=dict(range=r["x"]),
                        yaxis=dict(range=r["y"]),
                        zaxis=dict(range=r["z"]),
                        aspectmode="cube"
                    ))
                else:
                    r = ranges_by_kl.get(kl, dict(x=rx_all, y=ry_all))
                    layout_updates.update(xaxis=dict(range=r["x"]), yaxis=dict(range=r["y"]))
            buttons.append(dict(label=f"KL = {kl}", method="update",
                                args=[{"visible": vis_for_kl(kl)}, layout_updates]))
    elif buttons_by == "cluster":
        for c in clusters:
            layout_updates = {"title": f"{title} — Highlight Cluster {c}"}
            if compact and zoom_by == "cluster":
                if dim == 3:
                    r = ranges_by_cluster.get(c, dict(x=rx_all, y=ry_all, z=rz_all))
                    layout_updates.update(scene=dict(
                        xaxis=dict(range=r["x"]),
                        yaxis=dict(range=r["y"]),
                        zaxis=dict(range=r["z"]),
                        aspectmode="cube"
                    ))
                else:
                    r = ranges_by_cluster.get(c, dict(x=rx_all, y=ry_all))
                    layout_updates.update(xaxis=dict(range=r["x"]), yaxis=dict(range=r["y"]))
            label = "Noise" if c == -1 else f"Cluster {c}"
            buttons.append(dict(label=label, method="update",
                                args=[{"visible": vis_for_cluster(c)}, layout_updates]))
    else:
        raise ValueError("buttons_by must be 'kl' or 'cluster'")

    # Utility buttons
    buttons.append(dict(
        label="None", method="update",
        args=[{"visible": [True] + [False]*(total_traces-1)},
              {"title": f"{title} — Highlight: None"}]
    ))
    buttons.append(dict(
        label="All", method="update",
        args=[{"visible": [True] + [True]*(total_traces-1)},
              {"title": f"{title} — Highlight: All"}]
    ))

    # ---------- initial visibility: show first selection of buttons_by to get a legend ----------
    if buttons_by == "kl" and len(kl_values):
        initial_visible = vis_for_kl(kl_values[0])
        initial_title = f"{title} — Highlight KL={kl_values[0]}"
    elif buttons_by == "cluster" and len(clusters):
        initial_visible = vis_for_cluster(clusters[0])
        lab = "Noise" if clusters[0] == -1 else f"Cluster {clusters[0]}"
        initial_title = f"{title} — Highlight {lab}"
    else:
        initial_visible = [True] + [False]*(total_traces-1)
        initial_title = f"{title} — Highlight: None"

    for i, tr in enumerate(traces):
        tr.visible = initial_visible[i]

    # ---------- figure & layout ----------
    fig = go.Figure(data=traces)

    layout_kwargs = dict(
        title=initial_title,
        legend=dict(title=("Clusters" if legend_by=="cluster" else "KL-scores"),
                    itemclick="toggle", itemdoubleclick="toggleothers"),
        updatemenus=[dict(
            type="dropdown", direction="down", showactive=True,
            x=0.02, y=1.12, xanchor="left", yanchor="top",
            buttons=buttons
        )],
        margin=dict(l=30, r=30, t=80, b=30),
    )
    fig.update_layout(**layout_kwargs)

    if dim == 3:
        scene_dict = dict(
            xaxis_title="dim 0", yaxis_title="dim 1", zaxis_title="dim 2",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        )
        if compact:
            scene_dict.update(
                xaxis=dict(range=rx_all),
                yaxis=dict(range=ry_all),
                zaxis=dict(range=rz_all),
                aspectmode="cube"
            )
        fig.update_layout(scene=scene_dict)
    else:
        fig.update_xaxes(title="dim 0")
        fig.update_yaxes(title="dim 1")
        if compact:
            fig.update_xaxes(range=rx_all, constrain="domain")
            fig.update_yaxes(range=ry_all, constrain="domain")
            if equal_aspect:
                fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig