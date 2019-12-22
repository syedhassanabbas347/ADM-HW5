import pandas as pd
import networkx as nx
import time
import numpy as np
import matplotlib.pyplot as plt
import collections as cs
from collections import defaultdict
import random
import folium


class Graph():
    def __init__(func3):
        func3.edges = defaultdict(list)
        func3.weights = {}

    def add_edge(func3, from_node, to_node, weight):

        func3.edges[from_node].append(to_node)
        func3.weights[(from_node, to_node)] = weight


def position():
    position = pd.read_csv(r'C:\Users\aless\Desktop\HW5_ADM\data\position.co', header=None, sep=' ')
    position = position.drop([0], axis=1)
    position.columns = ['node', 'lat', 'long']
    return position


def distance_df():
    distance = pd.read_csv(r'C:\Users\aless\Desktop\HW5_ADM\data\distance.gr', header = None, sep = ' ')
    distance = distance.drop([0], axis=1)
    distance.columns = ['iniz', 'end', 'weight']
    return distance


def time_df():
    time_ = pd.read_csv(r'C:\Users\aless\Desktop\HW5_ADM\data\time.gr', header = None, sep = ' ')
    time_ = time_.drop([0], axis=1)
    time_.columns = ['iniz', 'end', 'weight']
    return time_


def net_dist():
    net_dist = pd.read_csv(r'C:\Users\aless\Desktop\HW5_ADM\data\time.gr', header = None, sep = ' ')
    net_dist = net_dist.drop([0, 3], axis=1)
    n = [1]*len(net_dist)
    net_dist['weight'] = n
    net_dist.columns = ['iniz', 'end', 'weight']
    return net_dist


def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    s = time.time()
    shortest_paths = {initial: (None, 0)}
    current_node = initial

    # set of visited nodes
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return 'Route not possible', np.inf
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []

    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node

    # Reverse path
    path = path[::-1]

    print('Dijkstra time: ', time.time()-s)
    return path, weight


# 2-OPT IMPLEMENTATION
def cost(cost_mat, route):
    #print(cost_mat)
    #print(route)
    return cost_mat[np.roll(route, 1), route].sum()  # shifts route array by 1 in order to look at pairs of cities


def two_opt(connect_mat, route):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue  # changes nothing, skip then
                new_route = route[:]  # Creates a copy of route
                new_route[i:j] = route[j - 1:i - 1:-1]  # this is the 2-optSwap since j >= i we use -1
                if cost(connect_mat, new_route) < cost(connect_mat, route):
                    route = new_route  # change current route to best
                    improved = True

        return route


def func_4():
    node = int(input('Enter your start node: '))

    route = list(map(int, input('Enter the nodes that you should visit: ').split()))

    print("\nChoose between the following weights:")
    print("1 - Time distance")
    print("2 - Physical distance")
    print("3 - Network distance")
    choice2 = int(input("Enter your choice: "))
    while choice2 not in [1, 2, 3]:
        print("Please, insert a valid choice!")
        choice2 = int(input("Enter your choice: "))

    if choice2 == 1:
        central_path_df, tot_path, build = func_4_time(node, route)
        frontend_f4(central_path_df, tot_path)
        frontend_f4_particular(build, tot_path, route)

    if choice2 == 2:
        central_path_df, tot_path, build = func_4_distance(node, route)
        frontend_f4(central_path_df, tot_path)
        frontend_f4_particular(build, tot_path, route)


    if choice2 == 3:
        central_path_df, tot_path, build = func_4_net(node, route)
        frontend_f4(central_path_df, tot_path)
        frontend_f4_particular(build, tot_path, route)


def func_4_time(node, route):
    print('Func_4_time begin')
    data = time_df()

    subset = data[['iniz', 'end', 'weight']]
    edges = [tuple(x) for x in subset.to_numpy()]
    graph = Graph()
    for edge in edges:
        graph.add_edge(*edge)

    print('Graph created')
    nodes = []
    for i in range(0, len(route) - 1):
        nodes.append(route[i])
    last = route[-1]
    return get_all(graph, nodes, node, last)


def func_4_distance(node, route):
    print('Func_4_time begin')
    data = distance_df()

    subset = data[['iniz', 'end', 'weight']]
    edges = [tuple(x) for x in subset.to_numpy()]
    graph = Graph()
    for edge in edges:
        graph.add_edge(*edge)

    print('Graph created')
    nodes = []
    for i in range(0, len(route) - 1):
        nodes.append(route[i])
    last = route[-1]
    return get_all(graph, nodes, node, last)


def func_4_net(node, route):
    print('Func_4_time begin')
    data = net_dist()

    subset = data[['iniz', 'end', 'weight']]
    edges = [tuple(x) for x in subset.to_numpy()]
    graph = Graph()
    for edge in edges:
        graph.add_edge(*edge)

    print('Graph created')
    nodes = []
    for i in range(0, len(route) - 1):
        nodes.append(route[i])
    last = route[-1]
    return get_all(graph, nodes, node, last)


def get_all(graph, nodes, node, last):
    adj_matrix = pd.DataFrame(np.matrix(np.ones((len(nodes), len(nodes))) * np.inf))
    path_matrix = []
    n = len(nodes)
    for i in range(0, n):
        path_m = []
        for j in range(0, n):
            if i == j:
                adj_matrix.iloc[i, j] = 0
                path_m.append(0)

            else:
                path, adj = dijsktra(graph, nodes[i], nodes[j])
                adj_matrix.iloc[i, j] = adj
                path_m.append(path)

        path_matrix.append(path_m)

    adj_matrix = adj_matrix.to_numpy()
    nodes = np.array(nodes)

    # GET INDEX FOR SMARTEST CENTRAL PATH
    giro = {}
    for i in range(0, len(nodes)):
        giro.update({i: nodes[i]})

    route = random.sample(range(len(nodes)), len(nodes))
    best = two_opt(adj_matrix, route)  # passing my adjacency matrix

    # PATH
    central = []
    for i in range(0, len(nodes)):
        central.append(giro.get(best[i], ))

    central_path = [[node, central[0]]]
    for i in range(0, len(central) - 1):
        central_path.append([central[i], central[i + 1]])
    central_path.append([central[-1], last])
    central_path_df = pd.DataFrame(central_path, columns=['iniz', 'end'])
    tot_path = [node]
    for i in range(0, len(central)):
        tot_path.append(central[i])
    tot_path.append(last)

    # CREATE PATH WITH ALL NODES
    first_to_first, wei = dijsktra(graph, node, central[0])
    whole_node = [first_to_first]
    for i in range(0, len(best) - 1):
        whole_node.append(path_matrix[best[i]][best[i + 1]])
    last_to_last, wei2 = dijsktra(graph, central[-1], last)
    whole_node.append(last_to_last)

    build = []
    for i in range(0, len(whole_node)):
        for j in range(0, len(whole_node[i]) - 1):
            build.append([whole_node[i][j], whole_node[i][j + 1]])

    build = pd.DataFrame(build, columns=['iniz', 'end'])

    print('Backend finish')
    return central_path_df, tot_path, build


def frontend_f4(build, route):
    posit = position()
    posi = dict([(i, (a, b)) for i, a, b in zip(posit.node, posit.long, posit.lat)])

    G = nx.from_pandas_edgelist(build, 'iniz', 'end')

    color_map = []
    lab = {}

    for node in G:
        color_map.append('blue')

        for k in range(0, len(route)):
            if node == route[k]:
                lab[node] = node
                color_map[-1] = 'green'

    color_map[0] = 'red'
    color_map[-1] = 'red'
    nx.draw(G, posi, node_color=color_map, with_labels=False)
    nx.draw_networkx_labels(G, posi, lab, font_size=16)
    plt.show()


def frontend_f4_particular(build, tot_path, route):
    posit = position()
    posi = dict([(i, (a, b)) for i, a, b in zip(posit.node, posit.long, posit.lat)])

    G = nx.from_pandas_edgelist(build, 'iniz', 'end')

    color_map = []
    lab = {}
    tp = []

    for node in G:
        color_map.append('blue')
        tp.append(node)

        for k in range(0, len(tot_path)):
            if node == tot_path[k]:
                lab[node] = node
                color_map[-1] = 'green'

    color_map[0] = 'red'
    color_map[-1] = 'red'
    nx.draw(G, posi, node_color=color_map, with_labels=False)
    nx.draw_networkx_labels(G, posi, lab, font_size=16)
    plt.show()

    # REAL MAP VISUALIZATION
    print('Real map start')

    locaz = ((posit.set_index('node')['long'][tot_path[0]]) / 1000000,
             (posit.set_index('node')['lat'][tot_path[0]]) / 1000000)
    m = folium.Map(location=locaz, zoom_start=10,
                   tiles='openstreetmap')

    i = 1
    for node in tp:
        folium.CircleMarker(location=((posit.set_index('node')['long'][node]) / 1000000,
                                      (posit.set_index('node')['lat'][node]) / 1000000), radius=3,
                            line_color='#3186cc', fill_color='#FFFFFF', fill_opacity=0.7, fill=True).add_to(m)
        for k in range(0, len(route)):
            if node == route[k]:
                folium.CircleMarker(location=((posit.set_index('node')['long'][node]) / 1000000,
                                              (posit.set_index('node')['lat'][node]) / 1000000), radius=7,
                                    line_color='green', fill_color='green', fill_opacity=0.7, fill=True,
                                    popup=('Passing '+ str(i) + ': ' + str(node))).add_to(m)
                i += 1

    folium.CircleMarker(location=((posit.set_index('node')['long'][tot_path[0]]) / 1000000,
                                  (posit.set_index('node')['lat'][tot_path[0]]) / 1000000), radius=7,
                        line_color='red', fill_color='red', fill_opacity=0.7, fill=True,
                        popup=('Start: ' + str(tot_path[0]))).add_to(m)
    folium.CircleMarker(location=((posit.set_index('node')['long'][tot_path[-1]]) / 1000000,
                                  (posit.set_index('node')['lat'][tot_path[-1]]) / 1000000), radius=7,
                        line_color='red', fill_color='red', fill_opacity=0.7, fill=True,
                        popup=('End: ' + str(tot_path[-1]))).add_to(m)

    m.save('frontend4part.html')
    print('frontend4part.html saved')




