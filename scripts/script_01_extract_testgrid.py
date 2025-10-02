import sys
import warnings
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pypsa
import networkx as nx

# add repo root
root_path = '../'
sys.path.append(root_path)

warnings.simplefilter(action='ignore', category=FutureWarning)

from utils.config import prenetwork_path, postnetwork_path
from utils.data_handling import (
    build_networkx_graph,
    get_effective_injections,
    get_matrices_from_nx_graph,
    sort_edge_indices,
    get_matrices_from_nx_graph_multigraph,
    get_effective_injections_multigraph,
    deaggregate_parallel_lines,
)

def extract_subnetwork(network, countries):
    network.determine_network_topology()
    max_count = 0
    max_index = np.inf

    for si, sub in enumerate(network.sub_networks.obj):        
        buses_in_subnet = sub.buses().index                
        n_tmp = network.copy()
        n_tmp.buses = n_tmp.buses.loc[buses_in_subnet]
        buses_in_countries = n_tmp.buses[n_tmp.buses.country.isin(countries)]
        count = len(buses_in_countries)
        if count > max_count:
            max_count = count
            max_index = si
            
    sub = network.sub_networks.obj[max_index]
    buses_in_subnet = sub.buses().index
    
    sub_net = network.copy()
    sub_net.buses = sub_net.buses.loc[buses_in_subnet]
    sub_net.generators = sub_net.generators[sub_net.generators.bus.isin(buses_in_subnet)]
    sub_net.loads = sub_net.loads[sub_net.loads.bus.isin(buses_in_subnet)]
    sub_net.lines = sub_net.lines[
        sub_net.lines.bus0.isin(buses_in_subnet) & sub_net.lines.bus1.isin(buses_in_subnet)
    ]
    sub_net.links = sub_net.links[
        sub_net.links.bus0.isin(buses_in_subnet) & sub_net.links.bus1.isin(buses_in_subnet)
    ]
    sub_net.storage_units = sub_net.storage_units[sub_net.storage_units.bus.isin(buses_in_subnet)]
    sub_net.stores = sub_net.stores[sub_net.stores.bus.isin(buses_in_subnet)]

    # add virtual loads for cut links
    original_links = network.links[~network.links.index.isin(sub_net.links.index)]
    cut_links = original_links[
        original_links.bus0.isin(buses_in_subnet) ^ original_links.bus1.isin(buses_in_subnet)
    ]

    for idx, link in cut_links.iterrows():
        inner_bus = link.bus0 if link.bus0 in buses_in_subnet else link.bus1
        outer_bus = link.bus1 if link.bus0 in buses_in_subnet else link.bus0
        sign = 1 if inner_bus == link.bus0 else -1
        sub_net.loads.loc[f"virtual_{idx}_{outer_bus}"] = {"bus": inner_bus, "name": f"virtual_{idx}_{outer_bus}"}
        sub_net.loads_t.p_set[f"virtual_{idx}_{outer_bus}"] = -sign * network.links_t.p0[idx]

    sub_net.mremove("Link", sub_net.links.index)
    sub_net.determine_network_topology()
    
    return sub_net

def main():
    # Sync grid selection
    sync_grid = "Scandinavia"  # options: "Scandinavia", "Great_Britain", "Continental_Europe"

    # country codes
    scandinavian_countries = ["SE", "NO", "FI", "DK"]
    gb_countries = ["GB"]
    ce_countries = [
        "AL", "AT", "BA", "BE", "BG", "CH", "CZ", "DE", "ES", "FR", "GR",
        "HR", "HU", "IT", "LU", "ME", "MK", "NL", "PL", "PT", "RO", "RS", "SI", "SK"
    ]

    # network paths
    paths = {
        "Scandinavia": prenetwork_path + "/Scandic_elec_s_900_ec_lv1.0_.nc",
        "Great_Britain": prenetwork_path + "/GB_elec_s_800_ec_lv1.0_.nc",
        "Continental_Europe": prenetwork_path + "/CE_elec_s_1200_ec_lv1.0_Co2L0.4.nc"
    }

    country_map = {
        "Scandinavia": scandinavian_countries,
        "Great_Britain": gb_countries,
        "Continental_Europe": ce_countries
    }

    countries = country_map[sync_grid]
    path_to_network = paths[sync_grid]

    # load network
    network = pypsa.Network(path_to_network)

    # extract subnetwork
    n_sync = extract_subnetwork(network, countries)
    n_sync.export_to_netcdf(postnetwork_path + f"{sync_grid}.nc")

    # build networkx graphs
    snapshot = n_sync.snapshots[0]
    G_sync = build_networkx_graph(n_sync, snet_index=0)

    # simple matrices
    P = get_effective_injections(n_sync, snapshot, G_sync)
    I_m, B_d, num_parallels, line_limits = get_matrices_from_nx_graph(G_sync)

    np.savez(postnetwork_path + f"{sync_grid}_simple_matrices.npz",
             P=P, I_m=I_m, B_d=B_d, num_parallels=num_parallels, line_limits=line_limits)

    with open(postnetwork_path + f"{sync_grid}_simple_graph.pkl", "wb") as f:
        pickle.dump(G_sync, f)

    # deaggregate parallel lines
    G_sync_deagg = deaggregate_parallel_lines(G_sync)
    G_sync_deagg = sort_edge_indices(G_sync_deagg)

    # deaggregated matrices
    P = get_effective_injections_multigraph(n_sync, snapshot, G_sync_deagg)
    I_m, B_d, num_parallels, line_limits = get_matrices_from_nx_graph_multigraph(G_sync_deagg)

    np.savez(postnetwork_path + f"{sync_grid}_deagg_matrices.npz",
             P=P, I_m=I_m, B_d=B_d, num_parallels=num_parallels, line_limits=line_limits)

    with open(postnetwork_path + f"{sync_grid}_deagg_graph.pkl", "wb") as f:
        pickle.dump(G_sync_deagg, f)

    # optional plot
    plt.figure(figsize=(10, 10))
    pos = nx.get_node_attributes(G_sync_deagg, 'pos')
    nx.draw(G_sync_deagg, pos=pos, node_size=12)
    plt.show()

if __name__ == "__main__":
    main()
