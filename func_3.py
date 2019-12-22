import pandas as pd
import networkx as nx
import time
import numpy as np
import matplotlib.pyplot as plt
import collections as cs
from collections import defaultdict
import folium


class Graph():
    def __init__(func3):
        func3.edges = defaultdict(list)
        func3.weights = {}

    def add_edge(func3, from_node, to_node, weight):

        func3.edges[from_node].append(to_node)
        # func3.edges[to_node].append(from_node)
        func3.weights[(from_node, to_node)] = weight
        # func3.weights[(to_node, from_node)] = weight


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
            return "Route Not Possible"
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
    return path


def func_3():
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
        sol = func_3_time(node, route)
        frontend_f3(sol, node, route)

    if choice2 == 2:
        sol = func_3_distance(node, route)
        frontend_f3(sol, node, route)

    if choice2 == 3:
        sol = func_3_net(node, route)
        frontend_f3(sol, node, route)


def func_3_time(node, route):
    print('Func_3_time begin')
    data = time_df()

    subset = data[['iniz', 'end', 'weight']]
    edges = [tuple(x) for x in subset.to_numpy()]
    graph = Graph()
    for edge in edges:
        graph.add_edge(*edge)

    print('Graph created')
    nodes = [node]
    for i in range(0, len(route)):
        nodes.append(route[i])

    trip = []
    for i in range(0, len(nodes) - 1):
        trip.append(dijsktra(graph, nodes[i], nodes[i + 1]))

    build = []
    for i in range(0, len(trip)):
        for j in range(0, len(trip[i]) - 1):
            build.append([trip[i][j], trip[i][j + 1]])

    build = pd.DataFrame(build, columns=['iniz', 'end'])
    print('Backend finish')
    return build


def func_3_distance(node, route):
    print('Func_3_distance begin')
    data = distance_df()

    subset = data[['iniz', 'end', 'weight']]
    edges = [tuple(x) for x in subset.to_numpy()]
    graph = Graph()
    for edge in edges:
        graph.add_edge(*edge)

    print('Graph created')
    nodes = [node]
    for i in range(0, len(route)):
        nodes.append(route[i])

    trip = []
    for i in range(0, len(nodes) - 1):
        trip.append(dijsktra(graph, nodes[i], nodes[i + 1]))

    build = []
    for i in range(0, len(trip)):
        for j in range(0, len(trip[i])-1):
            build.append([trip[i][j], trip[i][j+1]])

    build = pd.DataFrame(build, columns=['iniz', 'end'])
    print('Backend finish')
    return build


def func_3_net(node, route):
    print('Func_3_network begin')
    data = net_dist()

    subset = data[['iniz', 'end', 'weight']]
    edges = [tuple(x) for x in subset.to_numpy()]
    graph = Graph()
    for edge in edges:
        graph.add_edge(*edge)

    print('Graph created')
    nodes = [node]
    for i in range(0, len(route)):
        nodes.append(route[i])

    trip = []
    for i in range(0, len(nodes) - 1):
        trip.append(dijsktra(graph, nodes[i], nodes[i + 1]))

    build = []
    for i in range(0, len(trip)):
        for j in range(0, len(trip[i])-1):
            build.append([trip[i][j], trip[i][j+1]])

    build = pd.DataFrame(build, columns=['iniz', 'end'])
    print('Backend finish')
    return build


def frontend_f3(build, nod, route):
    posit = position()
    posi = dict([(i, (a, b)) for i, a, b in zip(posit.node, posit.long, posit.lat)])
    nodes = [nod]
    for i in range(0, len(route)):
        nodes.append(route[i])

    G = nx.from_pandas_edgelist(build, 'iniz', 'end')

    color_map = []
    tot_path = []
    lab = {}
    lab[nod] = nod
    for node in G:
        color_map.append('blue')
        tot_path.append(node)
        for k in range(0, len(route)):
            if node == route[k]:
                lab[node] = node
                color_map[-1] = 'green'

    color_map[0] = 'red'
    color_map[-1] = 'red'
    nx.draw(G, posi, node_color=color_map, with_labels=False)
    nx.draw_networkx_labels(G, posi, lab, font_size=16)
    plt.show()

    # REAL MAP VISUALIZATION: result avaible .html
    print('Real map start')
    locaz = ((posit.set_index('node')['long'][tot_path[0]]) / 1000000,
             (posit.set_index('node')['lat'][tot_path[0]]) / 1000000)
    m = folium.Map(location=locaz, zoom_start=10,
                   tiles='openstreetmap')

    i = 1
    for node in tot_path:
        folium.CircleMarker(location=((posit.set_index('node')['long'][node]) / 1000000,
                                      (posit.set_index('node')['lat'][node]) / 1000000), radius=3,
                            line_color='#3186cc', fill_color='#FFFFFF', fill_opacity=0.7, fill=True).add_to(m)
        for k in range(0, len(route)):
            if node == route[k]:
                folium.CircleMarker(location=((posit.set_index('node')['long'][node]) / 1000000,
                                              (posit.set_index('node')['lat'][node]) / 1000000), radius=7,
                                    line_color='green', fill_color='green', fill_opacity=0.7, fill=True,
                                    popup=('Passing ' + str(i) + ': ' + str(node))).add_to(m)
                i += 1

    folium.CircleMarker(location=((posit.set_index('node')['long'][tot_path[0]]) / 1000000,
                                  (posit.set_index('node')['lat'][tot_path[0]]) / 1000000), radius=7,
                        line_color='red', fill_color='red', fill_opacity=0.7, fill=True,
                        popup=('Start: ' + str(tot_path[0]))).add_to(m)
    folium.CircleMarker(location=((posit.set_index('node')['long'][tot_path[-1]]) / 1000000,
                                  (posit.set_index('node')['lat'][tot_path[-1]]) / 1000000), radius=7,
                        line_color='red', fill_color='red', fill_opacity=0.7, fill=True,
                        popup=('End: ' + str(tot_path[-1]))).add_to(m)

    m.save('frontend3.html')
    print('frontend3.html saved')





