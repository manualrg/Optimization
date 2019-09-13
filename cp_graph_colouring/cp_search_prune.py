
import networkx as nx
import numpy as np
import pandas as pd

#####################

def create_domain(max_number_of_colours, node_count):
    colours = np.arange(0, max_number_of_colours)
    colours_bank = [(x, 0) for x in range(0, max_number_of_colours)] # List of tuples (colour, n_times_used)
    matrix = np.tile(colours, node_count).reshape(node_count, max_number_of_colours)
    return matrix, colours_bank

def get_available_domain(v, domain):
    av_domain = list(domain[v, :])
    av_domain = list(filter(lambda x: x!= -1, av_domain))
    return av_domain

def set_node_colour(v, available_domain, solution, colours_bank_srt):
    available_colours = [x for x in colours_bank_srt if x[0] in available_domain]
    # Split in used and unused colours
    available_used_colours = []
    available_unused_colours = []
    for colour, times in available_colours:
        if times>0:
            available_used_colours.append((colour,times))
        else:
            available_unused_colours.append(((colour,times)))
    # Check if there are used colours
    if len(available_used_colours)>0:
        available_colours = [x[0] for x in available_used_colours] # are sorted by times used
    else:
        available_colours = [x[0] for x in available_unused_colours]
        available_colours.sort(key=lambda x: x) # min colour
    colour = available_colours[0]
    solution[v] = colour
    return colour, solution

def update_colour_bank(colours_bank, upd_colour):
    upd_colours_bank = []
    for c, n in colours_bank:
        if c == upd_colour:
            upd_n = n + 1
        else:
            upd_n = n
        upd_colours_bank.append((c, upd_n))
    return upd_colours_bank

def prune_domain(v_colour, v_neighbours, available_domain):
    for neighbour in v_neighbours:
        available_domain[neighbour, v_colour] = -1
    return available_domain

def compute_used_domain(domain):
    return np.sum(np.where(domain==-1, True, False), axis=1)

def highest_colour_used(colours_bank):
    return max([x[0] for x in colours_bank if x[1]>0])

def assign_colours(g, v, domain, solution, colours_bank, dev :bool = False):
    # Get node neighbours
    v_neighbours = list(g.neighbors(v))
    # Retrieve available domain by inspecting neighbours colours
    #v_available_domain = get_available_domain(v, v_neighbours, domain)
    v_available_domain = get_available_domain(v, domain)
    # Set current node colour from colours bank
    colours_bank_srt = sorted(colours_bank, key= lambda x: x[1], reverse=True)
    v_assinged_colour, solution = set_node_colour(v, v_available_domain, solution, colours_bank_srt)
    #print(v, v_neighbours, v_available_domain, v_assinged_colour)
    # Update colour bank
    colours_bank = update_colour_bank(colours_bank_srt, v_assinged_colour)
    # Update domain by setting to -1 current node colour in each neighbour
    domain = prune_domain(v_assinged_colour, v_neighbours, domain)
    #report.append(np.max(compute_used_domain(domain)))
    return v_assinged_colour, solution, colours_bank, domain


def dynamic_search(g: nx.Graph, nodes_df: pd.DataFrame, domain: np.ndarray, solution: list,
                   colours_bank: list, param_dyn_search: float = 0.2, dev: bool = False):
    """
    nodes_df: columns={"node_id": int, "pr": float, "used_domain_size": int}
    """
    n_rows = len(nodes_df)

    _nodes_df = nodes_df.copy()
    _nodes_df_srt_greedy = _nodes_df.sort_values(by=["pr"], ascending=[False]).copy()

    iterated = []
    to_iterate = _nodes_df.sort_values(by="pr", ascending=False)["node_id"].values

    it = 0
    it_hist = []
    n_colour_tracking = []
    end_condition = len(to_iterate) > 0

    while end_condition:
        v = to_iterate[0]  # pop first element
        to_iterate = to_iterate[1:]

        if dev:
            print("\n>>>> It: {}, node_id: {}".format(it, v))
            print(nodes_df[nodes_df["node_id"] == v])

        v_assinged_colour, solution, colours_bank, domain = \
            assign_colours(g, v, domain, solution, colours_bank, dev=False)

        if dev:
            print("v_assinged_colour: {}, post_node_domain: {}".format(v_assinged_colour, domain[v, :]))

        n_colour_tracking.append(highest_colour_used(colours_bank))

        # Set dynammic search strategy, on early stages use greedy approach
        # then get the nodes with the most exhausted domain (least colours left) first
        # A list whose index is a node and the value is the exhausted domain size
        used_domain_sizes = compute_used_domain(domain)
        # Conver to Series to update nodes_df["used_domain_size"]
        used_domain_pairs = pd.Series(index=range(0, n_rows), data=used_domain_sizes)
        _nodes_df["used_domain_size"] = used_domain_pairs
        flg_dynamic = (it / n_rows) > param_dyn_search
        if flg_dynamic:
            # Sort by dynamic search criterion and then by greedy search to de-ambiguate
            _nodes_df = _nodes_df.sort_values(by=["used_domain_size", "pr"], ascending=[False, False])
            # Filter nodes_df by to_iterate in order to get a new list of nodes sorted properly
            to_iterate_df = _nodes_df[_nodes_df["node_id"].isin(to_iterate)].copy()
        else:
            # Use pre-sorted greedy and filter
            to_iterate_df = _nodes_df_srt_greedy[_nodes_df_srt_greedy["node_id"].isin(to_iterate)].copy()

        to_iterate = to_iterate_df.index.values
        if dev:
            print("flg_dynamic: {}, to_iterate:\n{}".format(flg_dynamic, to_iterate_df[:3]))

        iterated.append(v)
        it += 1
        it_hist.append(it)
        end_condition = len(to_iterate) > 0

    data = np.array([it_hist, used_domain_pairs, n_colour_tracking]).T
    report_df = pd.DataFrame(data=data, columns=["iteration", "used_domain_size", "n_colours_used"])

    return solution, colours_bank, domain, report_df

def check_solution(solution, edges):
    solution_ko = []
    solution_ko_nodes = []
    for u, v in edges:
        v_colour = solution[u]
        u_colour = solution[v]
        flg_ko = (v_colour == u_colour)
        solution_ko.append(flg_ko)
        if flg_ko:
            solution_ko_nodes.append((u, v, v_colour))
    return solution_ko, solution_ko_nodes

def draw_graph(node_count, edges, solution):
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.Graph()
    G.add_nodes_from(range(node_count))
    G.add_edges_from(edges)

    pos = nx.spring_layout(G)

    nx.draw(G, pos, node_color=solution)
    nx.draw_networkx_labels(G, pos, {k: k for k in range(node_count)})

    plt.show()