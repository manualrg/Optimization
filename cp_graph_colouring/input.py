import networkx as nx
import pandas as pd

def create_graph(edges: list):
    g = nx.Graph()
    g.add_edges_from(edges)

    return g

def create_node_rank(g):
    pr = nx.pagerank(g, alpha=0.85)
    pr_list = []
    for v, pr_value in pr.items():
        pr_list.append((v, pr_value))
    pr_list_srt = sorted(pr_list, key=lambda x: x[1], reverse=True) # (v , pr)
    return pr_list_srt

def get_node_data(g, nodes):
    node_deg = list(g.degree(nodes))
    pr = nx.pagerank(g, alpha=0.85)
    data = []
    for v, deg in node_deg:
        pr_value = pr[v]
        data.append((v, pr_value, deg))
    df = pd.DataFrame(data=data, columns=["node_id", "pr", "degree"])
    df["used_domain_size"] = 0
    #nodes_df.index = nodes_df["node_id"]
    return df

def input_data(file_location :str):

    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    g = nx.Graph()
    g.add_edges_from(edges)
    nodes = sorted(list(g.nodes()))

    return edges, nodes, edge_count, node_count, g