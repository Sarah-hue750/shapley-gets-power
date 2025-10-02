import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import cm

from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
from utils.data_handling import build_networkx_graph, nx_edges_to_matrix_indices

from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


from utils.data_handling import (
    matrix_indices_to_nx_edges,
    remove_negligable_affected_lines,
)
from utils.utils import scale_fonts

import matplotlib as mpl

mpl.rcParams["text.usetex"] = False


def build_p0_graph(network, snapshot, p0_vals):  # @ToDo:
    """
    Build a plot graph from the network and snapshot.
    Assigns power flow values (p0) to edges and orients them according to flow direction.
    """
    G = build_networkx_graph(network)  # Build a NetworkX graph from the PyPSA network

    # Iterate over all edges to assign power flow and orientation
    for u, v in G.edges():
        idx = nx_edges_to_matrix_indices([(u, v)], G)[
            0
        ]  # Get matrix index for the edge
        p0 = p0_vals[idx]  # Get power flow value for this edge
        G[u][v]["p0"] = p0  # Assign power flow value
    return G  # Return the updated graph


def get_edge_colors_with_zero_highlight(
    G,
    edge_attribute="flow",
    cmap_name="cividis",
    vmin=None,
    vmax=None,
    zero_color="red",
    eps=1e-6,
):
    """
    Returns a list of edge colors for a networkx graph G.
    Edges with |edge_attribute| < eps are colored with `zero_color`.
    Others are colored using a LogNorm colormap.
    """
    values = np.array([abs(data[edge_attribute]) for _, _, data in G.edges(data=True)])
    values_nonzero = values[values > eps]

    if len(values_nonzero) == 0:
        raise ValueError("All edge values are zero or below eps; cannot apply LogNorm.")

    if vmin is None:
        vmin = values_nonzero.min()
    if vmax is None:
        vmax = values_nonzero.max()

    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    edge_colors = []
    for _, _, data in G.edges(data=True):
        val = abs(data[edge_attribute])
        if val < eps:
            edge_colors.append(zero_color)
        else:
            edge_colors.append(cmap(norm(val)))

    return edge_colors, norm, cmap


def create_motivational_layout():
    """
    Creates plot layout for motivational plots.

    Returns:
        fig (matplotlib figure)
        ax_orig (ax): ax where subgrid shll be displayed
        ax_map (ax): ax where total grid shall be displayed
        ax_dfe (ax): ax with power flow change due to outage of line e
        ax_dfl (ax): ax with power flow change due to outage of line l
        ax_dfle (ax): ax with power flow change due to outage of line e and l
        ax_phie (ax): ax with shap value of line e
        ax_phil (ax): ax with shap value of line l
        ax_phiel (ax): ax with sum of shap values of e and l
        cbar_ax (ax): colorbar ax for displayed flow changes/shap values
        cbar_map_ax (ax): colorbar ax for displayed grid maps
    """
    fig = plt.figure(figsize=(13, 6))

    cbar_map_ax = fig.add_axes([0.08, 0.15, 0.015, 0.7])

    outer_gs = gridspec.GridSpec(
        2, 4, width_ratios=[1, 1, 1, 1], wspace=0.05, hspace=0.2
    )

    # Left column: lower map with upper inset space
    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[:, 0], height_ratios=[1, 1], wspace=0.05, hspace=0.2
    )
    ax_orig = fig.add_subplot(gs_left[0], projection=ccrs.PlateCarree())
    ax_map = fig.add_subplot(gs_left[1], projection=ccrs.PlateCarree())
    # Right plot centered manually
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer_gs[:, 1:], height_ratios=[1, 1], width_ratios=[1, 1, 1]
    )
    ax_dfe = fig.add_subplot(gs_right[0, 0])
    ax_dfl = fig.add_subplot(gs_right[0, 1])
    ax_phie = fig.add_subplot(gs_right[1, 0])
    ax_phil = fig.add_subplot(gs_right[1, 1])

    ax_dfel = fig.add_subplot(gs_right[0, 2])  # middle cell → vertical centering
    ax_phiel = fig.add_subplot(gs_right[1, 2])

    # Add colorbar axes on the right side of figure (outside the gridspec)
    cbar_ax = fig.add_axes(
        [0.93, 0.15, 0.015, 0.7]
    )  # [left, bottom, width, height] in figure fraction

    # Titles
    # ax_map.set_title(r'$Sync Grid$', fontsize=12)
    ax_orig.set_title(r"$f_0$", fontsize=12)

    ax_dfe.set_title(r"$\Delta f (\{e\})$")
    ax_dfl.set_title(r"$\Delta f (\{l\})$")
    ax_dfel.set_title(r"$\Delta f (\{e, l\})$")
    ax_phie.set_title(r"$\phi^e$")
    ax_phil.set_title(r"$\phi^l$")
    ax_phiel.set_title(r"$\phi^{e} + \phi^{l}$")

    # Hide ticks
    for ax in [ax_map, ax_dfe, ax_dfl, ax_dfel, ax_phie, ax_phil, ax_phiel, ax_orig]:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 0.92, 1])  # Leave space for colorbar on right

    return (
        fig,
        ax_orig,
        ax_map,
        ax_dfe,
        ax_dfl,
        ax_dfel,
        ax_phie,
        ax_phil,
        ax_phiel,
        cbar_ax,
        cbar_map_ax,
    )


# TODO draw_labeled_multigraph needs refactoring to reduce code duplication
def draw_labeled_multigraph(
    G,
    ax=None,
    norm=None,
    cmap=None,
    arrows=False,
    node_kw={},
    offset_step=0.04,
    node_names=False,
    labeled_edges=[],  # List of (u, v, key) edges to label
    flow_format="{:.2f}",  # Format string for flow label
    label_offset=0,
    rem_edges=[],
    attr="flow",
    padding=0.2,
    G_big=None,
    alpha=0.1,
):
    """
    Draw directed MultiDiGraph with edges colored by 'flow' attribute.
    Parallel edges are drawn as offset straight lines.
    Near-zero flow edges are red and drawn without arrows.
    Only edges in `labeled_edges` receive flow labels.
    """

    if ax is None:
        fig, ax = plt.subplots()

    if node_kw is None:
        node_kw = {}

    if cmap is None:
        cmap = cm.cividis

    if type(rem_edges) == tuple:
        rem_edges = [rem_edges]

    # check if removed edges are in graph
    missing_edges = [e for e in rem_edges if e not in G.edges]
    if missing_edges:
        raise ValueError(f"Edges not found in graph: {missing_edges}")

    if G_big is not None:
        # Draw larger nodes for G_big
        pos_big = dict(G_big.nodes(data="pos"))
        nx.draw_networkx_nodes(G_big, pos=pos_big, ax=ax, alpha=alpha, **node_kw)
        # Group parallel edges between same node pair (undirected)
        edge_groups = defaultdict(list)
        for i, (u, v, key, data) in enumerate(G_big.edges(keys=True, data=True)):
            pair = tuple(sorted([u, v]))
            edge_groups[pair].append((i, u, v, key, data))

        # Iterate over grouped parallel edges
        for pair, edges in edge_groups.items():
            n = len(edges)
            offsets = np.linspace(
                -offset_step * (n - 1) / 2, offset_step * (n - 1) / 2, n
            )
            for (idx, u, v, key, data), offset in zip(edges, offsets):
                linestyle = "solid"  # Default linestyle
                color = "grey"
                src, tgt = (u, v)  # orientation doesnt matter
                arrowstyle = "-"

                # define edge start/end
                x1, y1 = pos_big[src]
                x2, y2 = pos_big[tgt]

                dx, dy = x2 - x1, y2 - y1
                length = np.hypot(dx, dy)
                if length == 0:
                    continue

                ox, oy = -dy / length * offset, dx / length * offset
                start = (x1 + ox, y1 + oy)
                end = (x2 + ox, y2 + oy)

                # draw edge indicator
                patch = FancyArrowPatch(
                    start,
                    end,
                    arrowstyle=arrowstyle,
                    color=color,
                    mutation_scale=10,
                    lw=1.5,
                    linestyle=linestyle,
                    alpha=alpha,
                )
                ax.add_patch(patch)
        ax.set_frame_on(True)

    # Extract node positions
    pos = dict(G.nodes(data="pos"))

    # --- Add padding around the graph ---
    x_vals, y_vals = zip(*pos.values())
    x_margin = (max(x_vals) - min(x_vals)) * padding
    y_margin = (max(y_vals) - min(y_vals)) * padding

    ax.set_xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
    ax.set_ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)

    # Get data limits and axis size in pixels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())

    # Draw nodes and optional node labels
    nx.draw_networkx_nodes(G, pos, ax=ax, **node_kw)
    if node_names:
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    # Group parallel edges between same node pair (undirected)
    edge_groups = defaultdict(list)
    for i, (u, v, key, data) in enumerate(G.edges(keys=True, data=True)):
        pair = tuple(sorted([u, v]))
        edge_groups[pair].append((i, u, v, key, data))

    # Iterate over grouped parallel edges
    for pair, edges in edge_groups.items():
        n = len(edges)
        offsets = np.linspace(-offset_step * (n - 1) / 2, offset_step * (n - 1) / 2, n)
        for (idx, u, v, key, data), offset in zip(edges, offsets):
            colored_value = data.get(attr)
            # print(f"Drawing edge {u}-{v} with key {key} and value {colored_value}")
            linestyle = "solid"  # Default linestyle

            # check arrow typs
            if arrows == True:
                u_ori, v_ori = data.get("orientation")
                src, tgt = (u_ori, v_ori) if colored_value > 0 else (v_ori, u_ori)
                arrowstyle = "-|>"
            else:
                src, tgt = (u, v)  # orientation doesnt matter
                arrowstyle = "-"

            # check if edge was removed
            if (u, v, key) in rem_edges:
                color = "black"
                arrowstyle = "-"
                linestyle = "dashed"
            elif (v, u, key) in rem_edges:
                color = "black"
                arrowstyle = "-"
                linestyle = "dashed"

            else:
                if attr == "flow":
                    color = cmap(norm(np.abs(colored_value)))
                else:
                    color = cmap(norm(colored_value))

            # define edge start/end
            x1, y1 = pos[src]
            x2, y2 = pos[tgt]

            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)
            if length == 0:
                continue

            ox, oy = -dy / length * offset, dx / length * offset
            start = (x1 + ox, y1 + oy)
            end = (x2 + ox, y2 + oy)

            # draw edge indicator
            patch = FancyArrowPatch(
                start,
                end,
                arrowstyle=arrowstyle,
                color=color,
                mutation_scale=10,
                lw=1.5,
                linestyle=linestyle,
            )
            ax.add_patch(patch)

            # Check for label condition (in either direction)
            edge_key = None
            if (u, v, key) in labeled_edges:
                edge_key = (u, v, key)
            elif (v, u, key) in labeled_edges:
                edge_key = (v, u, key)

            # Draw label if edge is in label list
            if edge_key is not None:

                # Midpoint of the edge
                mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2

                # Apply perpendicular offset (normal vector to the edge)
                kx = -(end[1] - start[1]) / length  # -dy / length
                ky = (end[0] - start[0]) / length  #  dx / length

                mx += label_offset * kx
                my += label_offset * ky

                # Compute angle for text rotation

                # Compute scale factors (data units per pixel)
                x_scale = (xlim[1] - xlim[0]) / bbox.width
                y_scale = (ylim[1] - ylim[0]) / bbox.height

                # Compute angle for text rotation
                dx_label, dy_label = x2 - x1, y2 - y1

                # Adjust dx, dy by aspect ratio
                dx_label = dx_label / x_scale
                dy_label = dy_label / y_scale

                # Calculate angle in degrees
                angle = np.degrees(np.arctan2(dy_label, dx_label))
                if angle > 90 or angle < -90:
                    angle += 180  # Flip upside-down text

                # Select label content
                if attr == "del_flow" and data["del_flow"] is not None:
                    data_to_show = data["del_flow"]
                    label = rf"$\Delta f_a = {flow_format.format(data_to_show)}$"
                elif attr == "load" and data["load"] is not None:
                    data_to_show = data["load"] * 100
                    label = rf"$\text{{load}} = {flow_format.format(data_to_show)}\%$"
                elif attr == "current" and data["current"] is not None:
                    data_to_show = data["current"]
                    label = rf"$\text{{I}} = {flow_format.format(data_to_show)} kA$"

                else:
                    data_to_show = data.get("flow", 0)
                    label = flow_format.format(np.abs(data_to_show))

                # Draw the text label
                ax.text(
                    mx,
                    my,
                    label,
                    fontsize=9,
                    ha="center",
                    va="center",
                    # backgroundcolor="white",
                    rotation=angle,
                    rotation_mode="anchor",
                )


def draw_labeled_multigraph_threshold(
    G,
    ax=None,
    norm=None,
    cmap=None,
    arrows=False,
    node_kw={},
    offset_step=0.04,
    node_names=False,
    labeled_edges=[],  # List of (u, v, key) edges to label
    flow_format="{:.2f}",  # Format string for flow label
    label_offset=0,
    rem_edges=[],
    label_rem_edges=False,
    attr="flow",
    padding=0.2,
    G_big=None,
    alpha=0.1,
    fontsize=9,
):
    """
    Draw directed MultiDiGraph with edges colored by 'flow' attribute.
    Parallel edges are drawn as offset straight lines.
    Near-zero flow edges are red and drawn without arrows.
    Only edges in `labeled_edges` receive flow labels.
    """

    if ax is None:
        fig, ax = plt.subplots()

    if node_kw is None:
        node_kw = {}

    if cmap is None:
        cmap = cm.cividis

    if type(rem_edges) == tuple:
        rem_edges = [rem_edges]

    # check if removed edges are in graph
    missing_edges = [e for e in rem_edges if e not in G.edges]
    if missing_edges:
        raise ValueError(f"Edges not found in graph: {missing_edges}")

    if G_big is not None:
        # Draw larger nodes for G_big
        pos_big = dict(G_big.nodes(data="pos"))
        nx.draw_networkx_nodes(G_big, pos=pos_big, ax=ax, alpha=alpha, **node_kw)
        # Group parallel edges between same node pair (undirected)
        edge_groups = defaultdict(list)
        for i, (u, v, key, data) in enumerate(G_big.edges(keys=True, data=True)):
            pair = tuple(sorted([u, v]))
            edge_groups[pair].append((i, u, v, key, data))

        # Iterate over grouped parallel edges
        for pair, edges in edge_groups.items():
            n = len(edges)
            offsets = np.linspace(
                -offset_step * (n - 1) / 2, offset_step * (n - 1) / 2, n
            )
            for (idx, u, v, key, data), offset in zip(edges, offsets):
                linestyle = "solid"  # Default linestyle
                color = "grey"
                src, tgt = (u, v)  # orientation doesnt matter
                arrowstyle = "-"

                # define edge start/end
                x1, y1 = pos_big[src]
                x2, y2 = pos_big[tgt]

                dx, dy = x2 - x1, y2 - y1
                length = np.hypot(dx, dy)
                if length == 0:
                    continue

                ox, oy = -dy / length * offset, dx / length * offset
                start = (x1 + ox, y1 + oy)
                end = (x2 + ox, y2 + oy)

                # draw edge indicator
                patch = FancyArrowPatch(
                    start,
                    end,
                    arrowstyle=arrowstyle,
                    color=color,
                    mutation_scale=10,
                    lw=1.5,
                    linestyle=linestyle,
                    alpha=alpha,
                )
                ax.add_patch(patch)
        ax.set_frame_on(True)

    # Extract node positions
    pos = dict(G.nodes(data="pos"))

    # --- Add padding around the graph ---
    x_vals, y_vals = zip(*pos.values())
    x_margin = (max(x_vals) - min(x_vals)) * padding
    y_margin = (max(y_vals) - min(y_vals)) * padding

    ax.set_xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
    ax.set_ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)

    # Get data limits and axis size in pixels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())

    # Draw nodes and optional node labels
    nx.draw_networkx_nodes(G, pos, ax=ax, **node_kw)
    if node_names:
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    # Group parallel edges between same node pair (undirected)
    edge_groups = defaultdict(list)
    for i, (u, v, key, data) in enumerate(G.edges(keys=True, data=True)):
        pair = tuple(sorted([u, v]))
        edge_groups[pair].append((i, u, v, key, data))

    # Iterate over grouped parallel edges
    for pair, edges in edge_groups.items():
        n = len(edges)
        offsets = np.linspace(-offset_step * (n - 1) / 2, offset_step * (n - 1) / 2, n)
        for (idx, u, v, key, data), offset in zip(edges, offsets):
            colored_value = data.get(attr)
            # print(f"Drawing edge {u}-{v} with key {key} and value {colored_value}")
            linestyle = "solid"  # Default linestyle

            # check arrow typs
            if arrows == True:
                u_ori, v_ori = data.get("orientation")
                src, tgt = (u_ori, v_ori) if colored_value > 0 else (v_ori, u_ori)
                arrowstyle = "-|>"
            else:
                src, tgt = (u, v)  # orientation doesnt matter
                arrowstyle = "-"

            # define edge start/end
            x1, y1 = pos[src]
            x2, y2 = pos[tgt]

            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)
            if length == 0:
                continue

            ox, oy = -dy / length * offset, dx / length * offset
            start = (x1 + ox, y1 + oy)
            end = (x2 + ox, y2 + oy)

            # check if edge was removed
            if (u, v, key) in rem_edges or (v, u, key) in rem_edges:
                color = "red"
                arrowstyle = "-"
                linestyle = "solid"
                if label_rem_edges and G_big is not None:
                    rem_edge = (u, v, key) if (u, v, key) in rem_edges else (v, u, key)
                    label = nx_edges_to_matrix_indices([rem_edge], G_big)[0]
                    # Midpoint of the edge
                    mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2

                    # Apply perpendicular offset (normal vector to the edge)
                    kx = -(end[1] - start[1]) / length  # -dy / length
                    ky = (end[0] - start[0]) / length  #  dx / length

                    mx += label_offset * kx
                    my += label_offset * ky

                    # Compute angle for text rotation

                    # Compute scale factors (data units per pixel)
                    x_scale = (xlim[1] - xlim[0]) / bbox.width
                    y_scale = (ylim[1] - ylim[0]) / bbox.height

                    # Compute angle for text rotation
                    dx_label, dy_label = x2 - x1, y2 - y1

                    # Adjust dx, dy by aspect ratio
                    dx_label = dx_label / x_scale
                    dy_label = dy_label / y_scale

                    # Calculate angle in degrees
                    angle = np.degrees(np.arctan2(dy_label, dx_label))
                    if angle > 90 or angle < -90:
                        angle += 180  # Flip upside-down text
                    ax.text(
                        mx,
                        my,
                        label,
                        fontsize=fontsize,
                        ha="center",
                        va="center",
                        # backgroundcolor="white",
                        rotation=angle,
                        rotation_mode="anchor",
                    )

            else:
                if attr == "flow":
                    color = cmap(norm(np.abs(colored_value)))
                else:
                    color = cmap(norm(colored_value))

            # draw edge indicator
            patch = FancyArrowPatch(
                start,
                end,
                arrowstyle=arrowstyle,
                color=color,
                mutation_scale=10,
                lw=1.5,
                linestyle=linestyle,
            )
            ax.add_patch(patch)

            # Check for label condition (in either direction)
            edge_key = None
            if (u, v, key) in labeled_edges:
                edge_key = (u, v, key)
            elif (v, u, key) in labeled_edges:
                edge_key = (v, u, key)

            # Draw label if edge is in label list
            if edge_key is not None:

                # Midpoint of the edge
                mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2

                # Apply perpendicular offset (normal vector to the edge)
                kx = -(end[1] - start[1]) / length  # -dy / length
                ky = (end[0] - start[0]) / length  #  dx / length

                mx += label_offset * kx
                my += label_offset * ky

                # Compute angle for text rotation

                # Compute scale factors (data units per pixel)
                x_scale = (xlim[1] - xlim[0]) / bbox.width
                y_scale = (ylim[1] - ylim[0]) / bbox.height

                # Compute angle for text rotation
                dx_label, dy_label = x2 - x1, y2 - y1

                # Adjust dx, dy by aspect ratio
                dx_label = dx_label / x_scale
                dy_label = dy_label / y_scale

                # Calculate angle in degrees
                angle = np.degrees(np.arctan2(dy_label, dx_label))
                if angle > 90 or angle < -90:
                    angle += 180  # Flip upside-down text

                # Select label content
                if attr == "del_flow" and data["del_flow"] is not None:
                    data_to_show = data["del_flow"]
                    label = rf"$\Delta = {flow_format.format(data_to_show)}$"
                elif attr == "load" and data["load"] is not None:
                    data_to_show = data["load"] * 100
                    label = rf"$\text{{load}} = {flow_format.format(data_to_show)}\%$"
                elif attr == "current" and data["current"] is not None:
                    data_to_show = data["current"]
                    label = rf"$\text{{I}} = {flow_format.format(data_to_show)} kA$"

                else:
                    data_to_show = data.get("flow", 0)
                    label = flow_format.format(np.abs(data_to_show))

                # Draw the text label
                ax.text(
                    mx,
                    my,
                    label,
                    fontsize=fontsize,
                    ha="center",
                    va="center",
                    # backgroundcolor="white",
                    rotation=angle,
                    rotation_mode="anchor",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        boxstyle="round,pad=0.2",
                        alpha=0.7,
                    ),
                )


def approximation_figure_compare_random(
    results,
    outage_lines,
    G_0,
    attr="flow_change",
    chosen_cluster_label=None,
    file_path=None,
    title=None,
    show_plot=None,
):
    """
    Creates figure to compare normal clustering with random clustering.

    Args:
        results (list): dict with results information
        outage_lines (list): list of indeces of outage lines
        G_0 (networkx Graph): original graph
        attr (string): attribute to consider. Default is "flow_change".
        chosen_cluster_label (float/int): Resolution of clustering to use. Default is None, which results in using the first of the list.
        file_path (string): Path where plot is saved. Default is None, for which it is not saved.
        title (string): Title for subplts. Default is None, which results in no title.
        show_plot (bool): Default is False. If plot is shown.
    """
    scale_fonts(0.75)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={"wspace": 0.3})
    if title is not None:
        axes[0, 1].set_title("Random")
        axes[0, 0].set_title(title)

    x_max, bin_edges = plot_error_frequency(
        results=results, chosen_label=chosen_cluster_label, ax=axes[1, 1], rand=True
    )
    norm_rand, cmap_rand = plot_approx_error_map(
        results=results,
        outage_lines=outage_lines,
        G_0=G_0,
        chosen_label=chosen_cluster_label,
        ax=axes[0, 1],
        fig=fig,
        rand=True,
    )
    # add colorbar
    axes[0, 1].plot([0, 1], [0, 1])
    divider2 = make_axes_locatable(axes[0, 1])

    plot_error_frequency(
        results=results,
        chosen_label=chosen_cluster_label,
        ax=axes[1, 0],
        x_max=x_max,
        bin_edges=bin_edges,
    )
    norm, cmap = plot_approx_error_map(
        results=results,
        outage_lines=outage_lines,
        G_0=G_0,
        chosen_label=chosen_cluster_label,
        ax=axes[0, 0],
        fig=fig,
        norm=norm_rand,
    )
    # add colorbar
    axes[0, 0].plot([0, 1], [0, 1])
    divider1 = make_axes_locatable(axes[0, 0])
    cax = divider1.append_axes("left", size="5%", pad=0.08)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    # cbar.formatter = ScalarFormatter()
    cbar.ax.yaxis.set_ticks_position("left")  # ticks on left
    cbar.ax.yaxis.set_label_position("left")  # label on left

    cax_rand = divider2.append_axes("left", size="5%", pad=0.08)
    cbar_rand = fig.colorbar(
        ScalarMappable(norm=norm_rand, cmap=cmap_rand), cax=cax_rand
    )
    # cbar_rand.formatter = ScalarFormatter()
    cbar_rand.ax.yaxis.set_ticks_position("left")  # ticks on left
    cbar_rand.ax.yaxis.set_label_position("left")  # label on left
    if attr == "flow_change":
        cbar.set_label(r"Approximation error (MW)", rotation=90, labelpad=2)
        cbar_rand.set_label(r"Approximation error (MW)", rotation=90, labelpad=2)
    elif attr == "current":
        cbar.set_label(r"Approximation error current (kA)", rotation=90, labelpad=2)
        cbar_rand.set_label(
            r"Approximation error current (kA)", rotation=90, labelpad=2
        )
    elif attr == "load":
        cbar.set_label(r"Approximation error load", rotation=90, labelpad=2)
        cbar_rand.set_label(r"Approximation error load", rotation=90, labelpad=2)

    if file_path is not None:
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()


def approximation_figure_compare_random_taylor(
    results,
    attr="flow_change",
    chosen_cluster_label=None,
    file_path=None,
    title=None,
    show_plot=None,
):
    """
    Creates figure to compare normal clustering with random clustering.

    Args:
        results (list): dict with results information
        outage_lines (list): list of indeces of outage lines
        attr (string): attribute to consider. Default is "flow_change".
        chosen_cluster_label (float/int): Resolution of clustering to use. Default is None, which results in using the first of the list.
        file_path (string): Path where plot is saved. Default is None, for which it is not saved.
        title (string): Title for subplts. Default is None, which results in no title.
        show_plot (bool): Default is False. If plot is shown.
    """
    scale_fonts(0.75)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"wspace": 0.3})
    if title is not None:
        axes[1].set_title("Random")
        axes[0].set_title(title)

    x_max, bin_edges = plot_error_frequency(
        results=results, chosen_label=chosen_cluster_label, ax=axes[1], rand=True
    )
    plot_error_frequency(
        results=results,
        chosen_label=chosen_cluster_label,
        ax=axes[0],
        x_max=x_max,
        bin_edges=bin_edges,
    )

    if file_path is not None:
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()


def plot_error_frequency(
    results,
    attr="flow_change",
    chosen_label=None,
    ax=None,
    rand=False,
    x_max=None,
    bin_edges=None,
):
    """
    Plot of approximation error frequency given a resuls dict.

    Args:
        results (dict): results dict with approximation errors
        attr (string): Attribute to plot. Default is "flow_change".
        chosen_label (label): resolution label of cluster dict approximation to display. Defaults is None, then the first is chosen.
        ax (ax): ax to plot onto. Default is None, then one is created.
        rand (bool): wether to plot the approximation or the corresponding random cluster. Defaults is False.
        x_max (float): x axis maximum value. Default is None, then just the default is taken.
        bin_edges (array): bin_edges to use in the histogramm. Default is None, then just 100 bins are taken.
    """
    if chosen_label is None:
        chosen_label = results["cluster_results"][0]["label"]
    else:
        for cluster_result in results["cluster_results"]:
            if cluster_result["label"] == chosen_label:
                if not rand:
                    abs_diff = cluster_result["differences"][f"abs_diff_{attr}"]
                else:
                    abs_diff = cluster_result["differences"][f"abs_rand_diff_{attr}"]
    values = {}
    for outage_line in abs_diff:
        for affected_line, val in abs_diff[outage_line].items():
            values[affected_line] = values.get(affected_line, 0) + val
    values = {k: abs(v) for k, v in values.items()}
    print(f"Max approx error: {max(values.values())}")
    # Plot histogram
    if ax is None:
        fig, ax = plt.subplots()
    if bin_edges is None:
        counts, bin_edges, patches = ax.hist(values.values(), bins=100)
    else:
        counts, bin_edges, patches = ax.hist(values.values(), bins=bin_edges)
    ax.set_xlabel("Approximation error (MW)")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    if x_max is None:
        x_max = ax.get_xlim()[1]
    ax.set_xlim(0, x_max)
    return x_max, bin_edges


def plot_approx_error_map(
    results,
    outage_lines,
    G_0,
    attr="flow_change",
    chosen_label=None,
    ax=None,
    fig=None,
    rand=False,
    norm=None,
):
    """
    Plots the approximation error of affected lines on a map.

    Args:
        results (dict): results dict with approximation errors
        outage_lines (list): list of indices of outage lines. They are dashed out in the plot.
        G_0 (networkx Graph): Original Graph.
        attr (string): Attribute to plot. Default is "flow_change".
        chosen_label (label): resolution label of cluster dict approximation to display. Defaults is None, then the first is chosen.
        ax (ax): ax to plot onto. Default is None, then one is created.
        fig (fig): Matplotlib figure to plot onto
        rand (bool): wether to plot the approximation or the corresponding random cluster. Defaults is False.
        norm (norm): norm for colorbar. Default is None, where the norm is determined by the max value.
    """
    if chosen_label is None:
        chosen_label = results["cluster_results"][0]["label"]
    else:
        for cluster_result in results["cluster_results"]:
            if cluster_result["label"] == chosen_label:
                if not rand:
                    abs_diff = cluster_result["differences"][f"abs_diff_{attr}"]
                else:
                    abs_diff = cluster_result["differences"][f"abs_rand_diff_{attr}"]
    values = {}
    for outage_line in abs_diff:
        for affected_line, val in abs_diff[outage_line].items():
            values[affected_line] = values.get(affected_line, 0) + val
    # Convert all sums to absolute
    values = {k: abs(v) for k, v in values.items()}
    shap_nx_edges = matrix_indices_to_nx_edges(values.keys(), G_0)
    shap_nx = {shap_nx_edges[idx]: v for idx, (k, v) in enumerate(values.items())}
    G_0_copy = G_0.copy()
    for edge in list(shap_nx_edges):
        G_0_copy.edges[edge]["del_flow"] = shap_nx[edge]
    if norm is None:
        norm = SymLogNorm(
            linthresh=1e-2,  # adjust depending on your data
            linscale=1.0,
            vmin=0,
            vmax=np.abs(list(values.values())).max(),
        )
    cmap = plt.get_cmap("Grays")
    NODE_ATTRS = {"node_color": "grey", "node_size": 2}
    draw_labeled_multigraph_threshold(
        G=G_0_copy,
        ax=ax,
        cmap=cmap,
        norm=norm,
        rem_edges=outage_lines,
        labeled_edges=[],
        label_offset=0,
        attr="del_flow",
        node_kw=NODE_ATTRS,
    )
    return norm, cmap


def plot_mult_error(
    results_dict,
    threshold=10,
    attr="flow_change",
    file_path=None,
    show_plot=False,
):
    """
    Plots different error measurements for multiple sets of outage lines (thus results_dict).
    Args:
        results_dict (list): list of results that contain approximation errrors for different outage line sets
        threshold (float): threshold after which flow changes are deemed significant in MW. Default is 10.
        attr (string): attr for which to plot approx errors. Default is "flow change".
        file_path (string): path to save plot to. Default is None, which results in not saving it.
        show_plot (bool): Default is False. If plot is shown.
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 10), sharex=True, gridspec_kw={"hspace": 0.2}
    )
    scale_fonts(0.75)
    num_above_threshold = {}
    total_num = {}
    box_plots = {}
    time_dict = {}
    shap_times = []
    maximum_approx_errors = {}
    for r in results_dict:
        results = r["results"]
        results_filtered = remove_negligable_affected_lines(
            results, {"flow_change": threshold, "load": 0.01, "current": 0.01}
        )
        shap_times.append(results["shap_time"])
        for cluster_result, cluster_result_filtered in zip(
            results["cluster_results"], results_filtered["cluster_results"]
        ):
            label = cluster_result["label"]
            time = cluster_result["approx_time"]
            time_dict.setdefault(label, []).append(
                abs(time) + r["cluster_creation_times"][label]
            )
            abs_diff = cluster_result["differences"][f"abs_diff_{attr}"]
            abs_diff_filtered = cluster_result_filtered["differences"][
                f"abs_diff_{attr}"
            ]
            values = {}
            values_filtered = {}
            for outage_line, outage_line_filtered in zip(abs_diff, abs_diff_filtered):
                for affected_line, val in abs_diff[outage_line].items():
                    values[affected_line] = values.get(affected_line, 0) + val
                for affected_line, val in abs_diff_filtered[
                    outage_line_filtered
                ].items():
                    if val is not None:
                        values_filtered[affected_line] = (
                            values_filtered.get(affected_line, 0) + val
                        )
            box_plots.setdefault(label, []).extend(
                abs(v) for v in values_filtered.values()
            )
            values = {k: abs(v) for k, v in values.items()}
            # maximum_approx_errors.setdefault(label, []).append(max(values.values()))
            maximum_approx_errors.setdefault(label, []).append(max(box_plots[label]))
            num_above_threshold[label] = num_above_threshold.get(label, 0) + sum(
                1 for v in values.values() if v > threshold
            )
            total_num[label] = total_num.get(label, 0) + sum(1 for v in values.values())
    maximum_approx_errors = {
        k: max(v) if v else None for k, v in maximum_approx_errors.items()
    }
    print("Maximum approx errors:", maximum_approx_errors)
    share_above_threshold = {
        k: num_above_threshold[k] / total_num[k] for k in num_above_threshold
    }
    means_time = [np.mean(time_dict[k]) for k in time_dict.keys()]
    stds_time = [
        np.std(time_dict[k], ddof=1) for k in time_dict.keys()
    ]  # ddof=1 for sample std
    sorted_box_plots = {k: box_plots[k] for k in sorted(box_plots.keys())}
    positions = np.array(list(sorted_box_plots.keys()))
    flierprops = dict(
        marker="o", markerfacecolor="black", markersize=1, linestyle="none"
    )

    axes[0].set_ylabel(r"$\sigma_a^{\text{approx}}$ (MW)")
    if max(list(box_plots.keys())) / min(list(box_plots.keys())) > 10:
        axes[0].set_xscale("log")
        widths = positions * 0.3
    else:
        widths = 0.2
    axes[0].set_yscale("log")
    axes[0].boxplot(
        sorted_box_plots.values(),
        tick_labels=sorted_box_plots.keys(),
        positions=positions,
        widths=widths,
        flierprops=flierprops,
    )

    # Sort keys
    sorted_keys = sorted(share_above_threshold.keys())
    sorted_share_above_threshold = [share_above_threshold[k] for k in sorted_keys]
    print("Share above threshold:", share_above_threshold)
    axes[1].plot(sorted_keys, sorted_share_above_threshold, marker="o")
    axes[1].set_ylabel(r"Share of $\sigma_a^{\text{approx}}$ > " + f"{threshold} MW")
    axes[2].errorbar(
        time_dict.keys(),
        means_time,
        yerr=stds_time,
        fmt="o",
        capsize=5,
        label="approximation",
    )
    print("Mean approx times:", {k: np.mean(v) for k, v in time_dict.items()})
    axes[2].axhline(
        np.mean(shap_times),
        color="black",
        linestyle="dashed",
        label="exact calculation",
    )
    axes[2].legend()
    axes[2].set_ylabel("Computation time (s)")
    axes[2].set_xlabel("k")

    if file_path is not None:
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()


def plot_waterfall(
    combined_effects,
    top_k=5,
    ax=None,
    file_path=None,
    f0=None,
    flip_sign=False,
    line_limit=None,
    ylabel="Flow change (MW)",
    label_size=10,
):
    """
    Plots a waterfall plot of all contributions to shapley taylor expansion.
    Args:
        combined_effects (dict): Dict of shap taylor expansion for one affected line
        top_k (int): Number of contributions that should be displayed, rest is in residual. Default is 5.
        ax (ax): Matplotlib ax to plot on. Default is None, which results in figure creation.
        file_path (string): File path, where to save the figure. Default is None, which results in no saving.
        f0 (float): Baseline value. Default is None, which results in no baseline being drawn.
        flip_sign (bool): Whether to flip the sign of all contributions. Default is False.
        line_limit (float): If given, a horizontal dashed line is drawn at this value. Default is None, which results in no line being drawn.
        ylabel (string): Y-axis label. Default is "Flow change (MW)".
        label_size (int): Font size of labels. Default is 10.
    """

    # Sort by absolute value (most important first)
    sorted_items_full = sorted(
        combined_effects.items(), key=lambda x: abs(x[1]), reverse=True
    )
    sorted_items = sorted_items_full[:top_k]
    residual = sum(v for _, v in sorted_items_full[top_k:])
    delta_f = sum(v for _, v in sorted_items_full)
    features, values = map(list, zip(*sorted_items))
    features.append("residual")
    values.append(residual)

    # Baseline and cumulative values
    baseline = 0
    if f0 is not None:
        baseline = f0
    positions = [baseline]
    for v in values:
        positions.append(positions[-1] + v)

    # Flip sign if needed, just change of direction of monitored line
    if flip_sign:
        baseline = -baseline
        values = [-v for v in values]
        delta_f = -delta_f

    # SHAP-like colors
    pink = (255 / 255, 137 / 255, 125 / 255)  # negative
    blue = (93 / 255, 163 / 255, 221 / 255)  # positive
    colors = [pink if v >= 0 else blue for v in values]

    # Plot waterfall
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    start = baseline
    if f0 is not None:
        ax.bar(-1, baseline, bottom=0, color="gray", edgecolor="black")
        ax.text(
            -1,
            baseline + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            "$f_0$",
            ha="center",
            va="bottom" if baseline >= 0 else "top",
            rotation=0,
            fontsize=label_size,
        )
        ax.plot(
            [-1, len(values)],
            [baseline, baseline],
            color="black",
            linestyle="--",
            linewidth=1,
        )

    for i, (f, v, c) in enumerate(zip(features, values, colors)):
        # Each bar starts at the current cumulative "start"
        ax.bar(i, v, bottom=start, color=c, edgecolor="black")

        # Draw a dashed horizontal line to the next bar (if not last)
        if i < len(values) - 1:
            ax.plot(
                [i, i + 1],
                [start + v, start + v],
                color="black",
                linestyle="--",
                linewidth=1,
            )

        # Update starting point
        start += v

    val_sum = 0
    for i, (label, val) in enumerate(
        zip(features + [f"$\\Delta f_a$"], values + [delta_f])
    ):
        val_sum += val
        if label == f"$\\Delta f_a$":
            val_sum = delta_f
        # Decide offset direction based on sign
        offset = 0.02 * (
            ax.get_ylim()[1] - ax.get_ylim()[0]
        )  # scale offset by y-axis range
        y_pos = val_sum + baseline + (offset if val >= 0 else -offset)

        ax.text(
            i,
            y_pos,
            label,
            ha="center",
            va="bottom" if val >= 0 else "top",
            rotation=0,
            bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.2"),
            fontsize=label_size,
        )

    # Formatting
    ax.bar(len(features), delta_f, bottom=baseline, color="gray", edgecolor="black")
    ax.plot(
        [len(features) - 1, len(features)],
        [delta_f + baseline, delta_f + baseline],
        color="black",
        linestyle="--",
        linewidth=1,
    )
    if line_limit is not None:
        ax.axhline(line_limit, color=pink, linestyle="dashed", label="line limit")
        ax.legend()

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position("zero")
    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    if file_path is not None:
        fig.savefig(file_path, dpi=300, bbox_inches="tight")


def plot_comparisons_third_order(shap_terms, ax=None, file_path=None):
    """
    Plots a comparison of all first vs. second vs. third order contributions of a third order shap expansion for all affected lines.
    Args:
        shap_terms (list): List of dicts of third order shap taylor expansioon for all affected lines
        ax (ax): Matplotlib ax to plot on. Default is None, which results in figure creation.
        file_path (string): File path, where to save the figure. Default is None, which results in no saving.
    """
    first_order_sum = []
    second_order_sum = []
    third_order_sum = []
    for i, shap_term in enumerate(shap_terms):
        first_order_sum.append(0)
        second_order_sum.append(0)
        third_order_sum.append(0)
        for k, v in shap_term["direct_effects"].items():
            first_order_sum[i] += v
        for k, v in shap_term["second_order_effects"].items():
            second_order_sum[i] += v
        for k, v in shap_term["third_order_shap"].items():
            third_order_sum[i] += v

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(np.abs(first_order_sum), label="1")
    ax.plot(np.abs(second_order_sum), label="2")
    ax.plot(np.abs(third_order_sum), label="3")
    ax.legend()
    ax.set_xlabel("Affected line")
    ax.set_ylabel("Absolute value of effects on affected lines (MW)")
    ax.set_yscale("log")
    if file_path is not None:
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()
