# --- Standard Library ---
import sys
import os
import pickle as pkl
import warnings



# --- Numerical Libraries ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.ticker import ScalarFormatter
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- Cartopy for map plotting ---
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- Environment setup ---
os.environ["OPENBLAS_NUM_THREADS"] = "1"
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)

# --- Project Config ---
from utils.config import (
    postnetwork_path,
    plots_path,
    braess_path,
    application_path,
)

# --- Utilities ---
from utils.utils import scale_fonts
from utils.data_handling import (
    add_flows_to_graph,
    nx_edges_to_matrix_indices,
    sort_edge_indices,
    get_flownetwork_without_lines,
    find_next_next_neighbors_until_connected,
    load_cases_csv,
    get_shap_network,
)
from utils.cascade_simulation import (
    solve_lpf,
)
from utils.plotting import (
    create_motivational_layout,
    draw_labeled_multigraph,
    draw_labeled_multigraph_threshold,
    plot_waterfall,
)
from utils.calculate_shap import (
    get_flow_attribution_shapley_value_all_affected_lines,
    get_shapley_taylor_all_outage_lines,
    get_all_direct_effects,
    flow_attribution_dict,
    get_flow_attribution_direct_effect_all_affected_lines,
    get_flow_attribution_shapley_taylor_all_affected_lines,
)
from scripts.script_03_find_application import load_quadruple_cases_csv

# --- Warnings ---
warnings.simplefilter(action='ignore', category=FutureWarning)

SYNC_GRID = "Scandinavia"  # Great_Britain, Scandinavia or Continental_Europe
SAVE_FILE_BRAESS = braess_path + f"braess_paradox_cases.csv"
SAVE_FILE_APPLICATIONS = application_path + f"application_cases.csv"






##### Plotting functions #####

def motivational_plot(
    G_0,
    G_multi_data,
    node_labels,
    rem_edges,
    labeled_edges,
    label_offset,
    label_offset_orig,
    ax_orig_aspect,
    ax_map_aspect,
    save_folder = plots_path,
    cmap = get_cmap("cividis"),
    cmap_delta = get_cmap("coolwarm"),
    attr="del_flow",
    NODE_ATTRS = {"node_color": "grey", "node_size": 50}, 
    
):
    """
    Create a motivational plot showing the flow network and the effects of removing specific edges.
    Args:
        G_0 (nx.MultiDiGraph): The original flow network graph.
        node_labels (list): List of node labels to include in the subgraph.
        rem_edges (list): List of edges to be removed for the analysis.
        labeled_edges (list): List of edges to be labeled in the plot.
        label_offset (float): Offset for edge labels.
        label_offset_orig (float): Offset for original edge labels.
        file_name (str): Name of the output file for the plot.
        ax_orig_aspect (float): Aspect ratio for the original graph axis.
        ax_map_aspect (float): Aspect ratio for the map axis.
    """
    (
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
    ) = create_motivational_layout()

    # Set aspect ratio for all axes
    for ax in [ax_orig, ax_dfe, ax_dfl, ax_dfel]:
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_aspect(1.5)  # Set aspect ratio to auto for all axes



    # export data from G_multi_data
    P = G_multi_data["P"]
    I_m = G_multi_data["I_m"].item()
    B_d = G_multi_data["B_d"].item()
    
    
    # subgraph with only edges of interest
    G_0_small = G_0.subgraph(node_labels).copy()

    
    #  removed cases
    G_e = get_flownetwork_without_lines(G_0, P, [rem_edges[0]])
    G_l = get_flownetwork_without_lines(G_0, P, [rem_edges[1]])
    G_el = get_flownetwork_without_lines(G_0, P, rem_edges)

    # subgraph with only the nodes of interest
    G_e_small = G_e.subgraph(node_labels).copy()
    G_l_small = G_l.subgraph(node_labels).copy()
    G_el_small = G_el.subgraph(node_labels).copy()

    # get Matrix values for shap calculation
    e_idx = nx_edges_to_matrix_indices([rem_edges[0]], G_0)[0]
    l_idx = nx_edges_to_matrix_indices([rem_edges[1]], G_0)[0]
    outage_lines_idx = [e_idx, l_idx]

    label_edge_G0 = [edge for edge in G_0.edges if edge in labeled_edges]
    label_edge_idx = nx_edges_to_matrix_indices(label_edge_G0, G_0)

    edges_G0_small = [
        edge for edge in G_0.edges if edge in G_0_small.edges
    ]  # to avoid other direction
    edges_idx = nx_edges_to_matrix_indices(edges_G0_small, G_0)
    affected_lines = [edge for edge in edges_idx if not edge in outage_lines_idx]

    ## Draw the original flows/currents/loads on the map
    attr_map = attr if attr != "del_flow" else "flow"
    # Collect attribute values from G_0 for these edges
    attr_values = [G_0_small.edges[edge][attr_map] for edge in edges_G0_small]

    # Convert to numpy array (optional)
    attr_array = np.array(attr_values)

    # Compute min and max
    vmin = attr_array.min()
    vmax = attr_array.max()
    if attr_map == "flow":
        norm = Normalize(
            vmin=abs(attr_array).min(), vmax=abs(attr_array).max(), clip=True
        )
    else:
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    # add colorbar
    cbar_map = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_map_ax)
    cbar_map.formatter = ScalarFormatter()
    cbar_map.formatter.set_scientific(False)
    cbar_map.formatter.set_powerlimits(
        (-2, 2)
    )  # Controls when to switch to sci notation
    cbar_map.update_ticks()
    if attr_map == "flow":
        cbar_map.set_label(r"Flow (MW)", rotation=90, labelpad=5)
    elif attr_map == "current":
        cbar_map.set_label(r"Current (kA)", rotation=90, labelpad=5)
    elif attr_map == "load":
        cbar_map.set_label(r"load (\\%)", rotation=90, labelpad=5)
    # Custom ticks

    if attr_map == "flow":
        step = 500
        candidate_ticks = np.arange(step, abs(attr_array).max(), step)
        middle_ticks = candidate_ticks[candidate_ticks >= abs(attr_array).min()]
        ticks = np.concatenate(
            ([abs(attr_array).min()], middle_ticks, [abs(attr_array).max()])
        )
        labels = (
            [f"≤{int(abs(attr_array).min())}"]
            + [f"{int(t)}" for t in middle_ticks]
            + [f"≥{int(abs(attr_array).max())}"]
        )

        # Set ticks and labels on colorbar
        cbar_map.set_ticks(ticks)
        cbar_map.set_ticklabels(labels)
    cbar_map.ax.yaxis.set_ticks_position("left")
    cbar_map.ax.yaxis.set_label_position("left")

    draw_labeled_multigraph(
        G_0,
        ax=ax_map,
        norm=norm,
        cmap=cmap,
        arrows=False,
        node_kw={"node_color": "grey", "node_size": 2},
        padding=0,
        attr=attr_map,
    )
    # add country borders
    ax_map.add_feature(cfeature.BORDERS, linestyle="-", linewidth=0.5)

    ## Draw original flows above map
    # in the case of del_flow plot orig flows, else plot current / loads
    attr_map = attr if attr != "del_flow" else "flow"
    draw_labeled_multigraph(
        G_0_small,
        ax=ax_orig,
        norm=norm,
        cmap=cmap,
        arrows=False,
        node_kw=NODE_ATTRS,
        labeled_edges=labeled_edges,
        label_offset=label_offset_orig,
        G_big=G_0,
        attr=attr_map,
    )

    ## Draw changed graphs and shapley attributions

    # Shap_e
    shap_e = get_flow_attribution_shapley_value_all_affected_lines(
        affected_lines=affected_lines,
        outage_line=e_idx,
        P=P,
        B_d=B_d,
        M=set(outage_lines_idx),
        I=I_m,
    )
    G_shap_e = get_shap_network(
        G_0=G_0, G_0_small=G_0_small, shap=shap_e, outage_lines=outage_lines_idx
    )

    # Shap l
    shap_l = get_flow_attribution_shapley_value_all_affected_lines(
        affected_lines=affected_lines,
        outage_line=l_idx,
        P=P,
        B_d=B_d,
        M=set(outage_lines_idx),
        I=I_m,
    )
    G_shap_l = get_shap_network(
        G_0=G_0, G_0_small=G_0_small, shap=shap_l, outage_lines=outage_lines_idx
    )

    # Shap e + Shap l
    shap_el = {k: shap_e[k] + shap_l[k] for k in shap_e.keys()}
    G_shap_el = get_shap_network(
        G_0=G_0, G_0_small=G_0_small, shap=shap_el, outage_lines=outage_lines_idx
    )

    values = [data[attr] for _, _, data in G_shap_el.edges(data=True) if attr in data]
    min_val = min(values)
    max_val = max(values)

    norm_delta = Normalize(
        vmin=-max([abs(min_val), abs(max_val)]), vmax=max([abs(min_val), abs(max_val)])
    )
    # plot Shap
    draw_labeled_multigraph(
        G_shap_e,
        ax=ax_phie,
        norm=norm_delta,
        cmap=cmap_delta,
        node_kw=NODE_ATTRS,
        rem_edges=rem_edges,
        labeled_edges=labeled_edges,
        label_offset=label_offset,
        attr=attr,
        G_big=G_0,
    )
    draw_labeled_multigraph(
        G_shap_l,
        ax=ax_phil,
        norm=norm_delta,
        cmap=cmap_delta,
        node_kw=NODE_ATTRS,
        rem_edges=rem_edges,
        labeled_edges=labeled_edges,
        label_offset=label_offset,
        attr=attr,
        G_big=G_0,
    )
    draw_labeled_multigraph(
        G_shap_el,
        ax=ax_phiel,
        norm=norm_delta,
        cmap=cmap_delta,
        node_kw=NODE_ATTRS,
        rem_edges=rem_edges,
        labeled_edges=labeled_edges,
        label_offset=label_offset,
        attr=attr,
        G_big=G_0,
    )

    # plot

    # if attr == "load":
    #     norm_delta = Normalize(vmin=0, vmax=1.5) # ratio of line powerflow and line capacity, defined for the upper plots!

    draw_labeled_multigraph(
        G_e_small,
        ax=ax_dfe,
        norm=norm_delta,
        cmap=cmap_delta,
        rem_edges=[rem_edges[0]],
        node_kw=NODE_ATTRS,
        labeled_edges=labeled_edges,
        label_offset=label_offset,
        attr=attr,
        G_big=G_0,
    )
    draw_labeled_multigraph(
        G_l_small,
        ax=ax_dfl,
        norm=norm_delta,
        cmap=cmap_delta,
        rem_edges=[rem_edges[1]],
        node_kw=NODE_ATTRS,
        labeled_edges=labeled_edges,
        label_offset=label_offset,
        attr=attr,
        G_big=G_0,
    )
    draw_labeled_multigraph(
        G_el_small,
        ax=ax_dfel,
        norm=norm_delta,
        cmap=cmap_delta,
        rem_edges=rem_edges,
        node_kw=NODE_ATTRS,
        labeled_edges=labeled_edges,
        label_offset=label_offset,
        attr=attr,
        G_big=G_0,
    )

    # add colorbar
    cbar = fig.colorbar(ScalarMappable(norm=norm_delta, cmap=cmap_delta), cax=cbar_ax)
    cbar.formatter = ScalarFormatter()
    cbar.formatter.set_scientific(False)
    cbar.formatter.set_powerlimits((-2, 2))  # Controls when to switch to sci notation
    cbar.update_ticks()
    if attr == "del_flow":
        cbar.set_label(r"$\Delta$ Flow (MW)", rotation=270, labelpad=5)

    elif attr == "load":
        ax_orig.set_title(r"$|\frac{f}{f_{\mathrm{max}}}|_{0}$")
        ax_dfe.set_title(r"$|\frac{f}{f_{\mathrm{max}}}|(\{e\})$")
        ax_dfl.set_title(r"$|\frac{f}{f_{\mathrm{max}}}|(\{l\})$")
        ax_dfel.set_title(r"|$\frac{f}{f_{\mathrm{max}}}|(\{e,l\})$")

        ax_phie.set_title(r"$|\frac{f}{f_{\mathrm{max}}}|(\vec{\phi}^{e})$")
        ax_phil.set_title(r"$|\frac{f}{f_{\mathrm{max}}}|(\vec{\phi}^{l})$")
        ax_phiel.set_title(
            r"$|\frac{f}{f_{\mathrm{max}}}|(\vec{\phi}^{e} + \vec{\phi}^{l})$"
        )

        cbar.set_label(r"Load (%)", rotation=270, labelpad=5)

    elif attr == "current":
        ax_orig.set_title(r"$I_{0}$")
        ax_dfe.set_title(r"$I(\{e\})$")
        ax_dfl.set_title(r"$I(\{l\})$")
        ax_dfel.set_title(r"$I(\{e,l\})$")

        ax_phie.set_title(r"$I(\vec{\phi}^{e})$")
        ax_phil.set_title(r"$I(\vec{\phi}^{l})$")
        ax_phiel.set_title(r"$I(\vec{\phi}^{e} + \vec{\phi}^{l})$")

        cbar.set_label(r"Current (kA)", rotation=270, labelpad=5)

    # add rectangle to the map
    x_min, x_max = ax_orig.get_xlim()
    y_min, y_max = ax_orig.get_ylim()
    rect = plt.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        transform=ccrs.PlateCarree(),
        color="none",
        ec="black",
        lw=1.5,
        zorder=10,
    )
    ax_map.add_patch(rect)

    # set aspect ratio for all axes and remove ticks
    for ax in [ax_orig, ax_map]:
        ax.set_xticks([])
        ax.set_yticks([])
    ax_orig.set_aspect(ax_orig_aspect)
    ax_map.set_aspect(ax_map_aspect)

    fig.savefig(f"{save_folder}/motivational_plot.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{save_folder}/motivational_plot.pdf", dpi=300, bbox_inches="tight")

def interaction_equation_plot(
    G_0, G_multi_data, rem_edges, labeled_edges, label_offset, node_labels,save_folder = plots_path):
    
    # export data from G_multi_data
    P = G_multi_data["P"]
    I_m = G_multi_data["I_m"].item()
    B_d = G_multi_data["B_d"].item()
    
    # Norm and colormap for delta flows
    rc("mathtext", fontset="dejavuserif")
    cmap_delta = get_cmap("coolwarm")
    NODE_ATTRS = {"node_color": "grey", "node_size": 50}
    fig, ax = plt.subplots(1, 4, figsize=(16, 5))

    G_0_small = G_0.subgraph(node_labels).copy()

    # # get Matrix values for shap calculation
    e_idx = nx_edges_to_matrix_indices([rem_edges[0]], G_0)[0]
    l_idx = nx_edges_to_matrix_indices([rem_edges[1]], G_0)[0]
    outage_lines_idx = [e_idx, l_idx]
    edges_G0_small = [
        edge for edge in G_0.edges if edge in G_0_small.edges
    ]  # to avoid other direction
    edges_idx = nx_edges_to_matrix_indices(edges_G0_small, G_0)
    affected_lines = [edge for edge in edges_idx if not edge in outage_lines_idx]

    direct_effect_e = get_flow_attribution_direct_effect_all_affected_lines(
        affected_lines=affected_lines, outage_line=e_idx, P=P, B_d=B_d, I=I_m
    )
    G_direct_effect_e = get_shap_network(
        G_0=G_0,
        G_0_small=G_0_small,
        shap=direct_effect_e,
        outage_lines=outage_lines_idx,
    )

    direct_effect_l = get_flow_attribution_direct_effect_all_affected_lines(
        affected_lines=affected_lines, outage_line=l_idx, P=P, B_d=B_d, I=I_m
    )
    G_direct_effect_l = get_shap_network(
        G_0=G_0,
        G_0_small=G_0_small,
        shap=direct_effect_l,
        outage_lines=outage_lines_idx,
    )

    interaction_effect = get_flow_attribution_shapley_taylor_all_affected_lines(
        affected_lines=affected_lines,
        outage_line_1=e_idx,
        outage_line_2=l_idx,
        P=P,
        B_d=B_d,
        M=set(outage_lines_idx),
        I=I_m,
    )
    G_interaction_effect = get_shap_network(
        G_0=G_0,
        G_0_small=G_0_small,
        shap=interaction_effect,
        outage_lines=outage_lines_idx,
    )

    filtered_effects = {
        k: direct_effect_e[k] for k in edges_idx if k in direct_effect_e
    }
    norm_delta = SymLogNorm(
        linthresh=1e-2,  # adjust depending on your data
        linscale=1.0,
        vmin=-np.abs(list(filtered_effects.values())).max(),
        vmax=np.abs(list(filtered_effects.values())).max(),
    )
    norm_delta = Normalize(
        vmin=-2 * np.abs(list(filtered_effects.values())).max(),
        vmax=2 * np.abs(list(filtered_effects.values())).max(),
    )

    #   left panel: direct effect of a
    draw_labeled_multigraph(
        G_direct_effect_e,
        ax=ax[0],
        norm=norm_delta,
        cmap=cmap_delta,
        node_kw=NODE_ATTRS,
        rem_edges=rem_edges[0],
        labeled_edges=labeled_edges,
        label_offset=label_offset,
        attr="del_flow",
        G_big=G_0,
    )
    # right panel: direct effect of b
    ax[0].set_title(r"$\Delta f(\{e\})$", fontsize=20)
    draw_labeled_multigraph(
        G_direct_effect_l,
        ax=ax[1],
        norm=norm_delta,
        cmap=cmap_delta,
        node_kw=NODE_ATTRS,
        rem_edges=rem_edges[1],
        labeled_edges=labeled_edges,
        label_offset=label_offset,
        attr="del_flow",
        G_big=G_0,
    )
    ax[1].set_title(r"$\Delta f(\{l\})$", fontsize=20)
    #   middle panel: interaction effect
    draw_labeled_multigraph(
        G_interaction_effect,
        ax=ax[2],
        norm=norm_delta,
        cmap=cmap_delta,
        node_kw=NODE_ATTRS,
        rem_edges=rem_edges,
        labeled_edges=labeled_edges,
        label_offset=label_offset,
        attr="del_flow",
        G_big=G_0,
    )
    ax[2].set_title(r"$\iota_{el}$", fontsize=20)

    # panel 4: total flow change
    G_el = get_flownetwork_without_lines(G_0, P, rem_edges)
    G_el_small = G_el.subgraph(node_labels).copy()
    draw_labeled_multigraph(
        G_el_small,
        ax=ax[3],
        norm=norm_delta,
        cmap=cmap_delta,
        rem_edges=rem_edges,
        node_kw=NODE_ATTRS,
        labeled_edges=labeled_edges,
        label_offset=label_offset,
        attr="del_flow",
        G_big=G_0,
    )
    ax[3].set_title(r"$\Delta f(\{e,l\})$", fontsize=20)

    plus_positions = [0.31, 0.5125]  # Between subplot 0 & 1, and 1 & 2
    equal_position = 0.715  # Between subplot 2 & 3

    # Add plus signs
    for x in plus_positions:
        fig.text(x, 0.5, "+", fontsize=30, fontweight="bold", ha="center", va="center")

    # Add equal sign
    fig.text(
        equal_position,
        0.5,
        "=",
        fontsize=30,
        fontweight="bold",
        ha="center",
        va="center",
    )

    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # # add colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ScalarMappable(norm=norm_delta, cmap=cmap_delta), cax=cax)
    cbar.set_label(r"$\Delta f$ (MW)", rotation=270, labelpad=5)

    fig.savefig(
        f"{save_folder}/interaction_plot_example.pdf", dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        f"{save_folder}/interaction_plot_example.png", dpi=300, bbox_inches="tight"
    )
    
def application_plot(G_0, G_multi_data, 
    cmap = get_cmap("cividis"),
    NODE_ATTRS = {"node_color": "grey", "node_size": 20},
    file_name = SAVE_FILE_APPLICATIONS,
    save_folder = plots_path,
    ):
    
    """
    Create application plots for specific outage cases.
    Args:
        G_0 (nx.MultiDiGraph): The original flow network graph.
        G_multi_data (dict): Dictionary containing matrices and data for the flow network.
        cmap (matplotlib colormap): Colormap for plotting.
        NODE_ATTRS (dict): Node attributes for plotting.
    """
    
    
    # export data from G_multi_data
    P = G_multi_data["P"]
    I_m = G_multi_data["I_m"].item()
    B_d = G_multi_data["B_d"].item()
    # Load application cases


    scale_fonts(1)


    cases = load_quadruple_cases_csv(file_name)
    print(len(cases), "cases found")
    for case in cases[2:3]:
        # Load lines from case

        non_outage_line = case[5][0]  # line for which outage should be reversed

        outage_lines = case[0:4]  # all outage lines

        normal_outage_lines = [
            edge for edge in outage_lines if edge != non_outage_line
        ]  # outage lines excluding the non-outage line

        affected_line = case[4][0]  # line for which flow change should be explained

        # get indices of lines

        non_outage_line_idx = nx_edges_to_matrix_indices([non_outage_line], G_0)[0]

        affected_line_idx = nx_edges_to_matrix_indices([affected_line], G_0)[0]

        outage_lines_idx = nx_edges_to_matrix_indices(outage_lines, G_0)

        # get shapley-taylor values and direct effects

        direct_effects = get_all_direct_effects(B_d=B_d, M=set(outage_lines_idx), I=I_m)

        phi_taylor = get_shapley_taylor_all_outage_lines(
            B_d=B_d, M=set(outage_lines_idx), I=I_m
        )
        # attribute them to the affected line

        shap_taylor = flow_attribution_dict(
            affected_line=affected_line_idx, B_d=B_d, P=P, I=I_m, value_dict=phi_taylor
        )

        direct_effect_attr = flow_attribution_dict(
            affected_line=affected_line_idx, B_d=B_d, P=P, I=I_m, value_dict=direct_effects
        )

        # combine both effects

        combined_effects = {}

        for k, v in shap_taylor.items():

            combined_effects[k] = v

        for k, v in direct_effect_attr.items():

            combined_effects[k] = v

        # Get flownetwork after outages

        G_outages = get_flownetwork_without_lines(G_0, P, list(outage_lines))

        nodes_subgraph = find_next_next_neighbors_until_connected(
            G_0, list(outage_lines) + [affected_line]
        )

        G_0_small = G_0.subgraph(nodes_subgraph).copy()

        G_outages_small = G_outages.subgraph(nodes_subgraph).copy()

        # Get baseline and affected flow

        f_0 = G_0.edges[affected_line]["flow"]

        f_a = G_outages.edges[affected_line]["flow"]

        line_limit = G_0.edges[affected_line]["s_nom"]

        flip_sign = f_a < 0

        # Plot waterfall

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        plot_waterfall(
            combined_effects=combined_effects,
            ax=axes[1],
            f0=f_0,
            flip_sign=flip_sign,
            line_limit=line_limit,
            ylabel="Power flow (MW)",
            label_size=12,
        )

        # Get norm for colorbar

        attr_map = "load"

        attr_values = [
            G_outages_small.edges[edge][attr_map] for edge in G_outages_small.edges
        ]

        attr_array = np.array(attr_values)

        vmin = attr_array.min()

        vmax = attr_array.max()

        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

        # Plot network

        draw_labeled_multigraph_threshold(
            G_outages,
            ax=axes[0],
            norm=norm,
            cmap=cmap,
            padding=0.05,
            node_kw=NODE_ATTRS,
            rem_edges=list(outage_lines),
            labeled_edges=[affected_line],
            label_offset=0.4,
            G_big=G_outages,
            attr=attr_map,
            label_rem_edges=True,
            fontsize=12,
        )
        # colorbar

        divider1 = make_axes_locatable(axes[0])

        cax = divider1.append_axes("left", size="5%", pad=0.08)

        cbar_map = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)

        cbar_map.formatter = ScalarFormatter()

        cbar_map.formatter.set_scientific(True)

        cbar_map.formatter.set_powerlimits(
            (-2, 2)
        )  # Controls when to switch to sci notation

        cbar_map.update_ticks()

        if attr_map == "flow":

            cbar_map.set_label(r"Flow (MW)", rotation=90, labelpad=5)

        elif attr_map == "current":

            cbar_map.set_label(r"Current (kA)", rotation=90, labelpad=5)

        elif attr_map == "load":

            cbar_map.set_label(r"Power flow (% of line limit)", rotation=90, labelpad=5)

        cbar_map.ax.yaxis.set_ticks_position("left")

        cbar_map.ax.yaxis.set_label_position("left")

        # Show and save

        plt.show()

        fig.savefig(f"{save_folder}/application_quadruple_seed46_3_horizontal.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(f"{save_folder}/application_quadruple_seed46_3_horizontal.png", bbox_inches="tight", dpi=300)


def main():
    
    
    
    print("Creating plots...")
    
    if not os.path.exists(plots_path):
        print(f"Creating directory {plots_path}...")
        os.makedirs(plots_path)
    else:
        print(f"Directory {plots_path} exists.")

    print("Creating comparison plot between lodfs and shapley values...")


    # read pickled graph
    
    # check if file exists
    if not os.path.exists(f"{postnetwork_path}/{SYNC_GRID}_deagg_graph.pkl"):
        raise FileNotFoundError(f"File {postnetwork_path}/{SYNC_GRID}_deagg_graph.pkl not found.")
    if not os.path.exists(f"{postnetwork_path}/{SYNC_GRID}_deagg_matrices.npz"):
        raise FileNotFoundError(f"File {postnetwork_path}/{SYNC_GRID}_deagg_matrices.npz not found.")
    
    G_multi = pkl.load(open(f"{postnetwork_path}/{SYNC_GRID}_deagg_graph.pkl", "rb"))
    G_multi = sort_edge_indices(G_multi)
    G_multi_data = np.load(
        f"{postnetwork_path}/{SYNC_GRID}_deagg_matrices.npz", allow_pickle=True
    )
    P = G_multi_data["P"]
    I_m = G_multi_data["I_m"].item()
    B_d = G_multi_data["B_d"].item()

    # calculate undisrupted network
    flows_0 = solve_lpf(P=P, B_d=B_d, I=I_m)
    G_0 = add_flows_to_graph(G_multi, flows_0)

   

    print(f"Loading Braess paradox cases from {SAVE_FILE_BRAESS}...")
    
    # check if file exists
    if not os.path.exists(SAVE_FILE_BRAESS):
        raise FileNotFoundError(f"File {SAVE_FILE_BRAESS} not found.")
    
    cases = load_cases_csv(SAVE_FILE_BRAESS)

    cases_country = []
    prefix = "FI"
    for c in cases:
        e1, e2, reversed_edges = c
        if (
            e1[0].startswith(prefix)
            and e2[0].startswith(prefix)
            and reversed_edges[0][0].startswith(prefix)
        ):
            cases_country.append(c)

    # find the unique e1 e2 combis
    cases = cases_country
    cases_unique = set()
    for c in cases_country:
        e1, e2, reversed_edges = c
        cases_unique.add((e1, e2))
        
        
    # choose example no. 3
    idx = 3

    #  draw removed cases
    rem_edges = [cases[idx][0], cases[idx][1]]  # remove line FI2 5 - FI2 24

    nodes_subgraph = find_next_next_neighbors_until_connected(G_0, rem_edges)
    node_labels = nodes_subgraph  # [f"{prefix}{node}" for node in nodes]
    labeled_edges = cases[idx][2]
    label_offset = -0.23
    label_offset_orig = 0.23
    ax_orig_aspect = 1.5
    ax_map_aspect = 1.75

    # motivational plot
    print(f"Creating motivational plot...")
    motivational_plot(
        G_0=G_0,
        G_multi_data = G_multi_data,
        node_labels = node_labels,
        rem_edges = rem_edges,
        labeled_edges = labeled_edges,
        label_offset = label_offset,
        label_offset_orig = label_offset_orig,
        ax_orig_aspect=ax_orig_aspect,
        ax_map_aspect=ax_map_aspect,
    )

    # interaction equation plot
    print(f"Creating interaction equation plot...")
    interaction_equation_plot(
        G_0 = G_0,
        G_multi_data = G_multi_data,
        rem_edges = rem_edges,
        labeled_edges = labeled_edges,
        label_offset = label_offset,
        node_labels = node_labels,
    )
    
    
    # check if application file exists
    if not os.path.exists(SAVE_FILE_APPLICATIONS):
        raise FileNotFoundError(f"File {SAVE_FILE_APPLICATIONS} not found.")
    # check if application file is not empty
    print(f"Loading application cases from {SAVE_FILE_APPLICATIONS}...")
    # application plot
    print("Creating application plot...")
    application_plot(G_0=G_0, G_multi_data = G_multi_data)
        
    
        
if __name__ == "__main__":
    main()