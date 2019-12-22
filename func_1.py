import pandas as pd
import networkx as nx
import time
import numpy as np
import matplotlib.pyplot as plt
import folium


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


def visiting(data, node, th, graph):
    for i in range(0, len(data)):
        if data.iloc[i, 0] == node:
            if data.iloc[i, 2] <= th:
                print('Node ok')
                graph.append([data.iloc[i, 0], data.iloc[i, 1], data.iloc[i, 2]])

                # drop node visited
                data2 = data.drop(data.iniz[data.iniz == data.iloc[i, 0]].index.tolist())
                data2 = data2.drop(data.end[data.end == data.iloc[i, 0]].index.tolist())
                # RECURSIVE PART
                visiting(data2, data.iloc[i, 1], (th - data.iloc[i, 2]), graph)
    return graph


def func_1():

    node = int(input('Enter your node: '))

    print("\nChoose between the following weights:")
    print("1 - Time distance")
    print("2 - Physical distance")
    print("3 - Network distance")
    choice2 = int(input("Enter your choice: "))
    while choice2 not in [1, 2, 3]:
        print("Please, insert a valid choice!")
        choice2 = int(input("Enter your choice: "))

    th = int(input('Enter your threshold: '))

    if choice2 == 1:
        sol = func_1_time(node, th)
        frontend_f1_time(sol)

    if choice2 == 2:
        sol = func_1_distance(node, th)
        frontend_f1_distance(sol)

    if choice2 == 3:
        sol = func_1_net(node, th)
        frontend_f1_net(sol)


def func_1_time(node, th):
    print('Func_1_time begin')
    s = time.time()
    data = time_df()
    g = []
    graph = visiting(data, node, th, g)
    graph = pd.DataFrame(graph)
    print(graph)
    graph.columns = ['iniz', 'end', 'weight']
    print('Backend finish whit time:', time.time()-s)

    return graph


def frontend_f1_time(sol):
    posit = position()
    posi = dict([(i, (a, b)) for i, a, b in zip(posit.node, posit.long, posit.lat)])
    G = nx.from_pandas_edgelist(sol, 'iniz', 'end', 'weight')

    color_map = []
    tot_path = []
    for node in G:
        color_map.append('green')
        tot_path.append(node)
    color_map[0] = 'red'
    nx.draw(G, posi, node_color=color_map, with_labels=True, edge_color='b')

    plt.show()

    # REAL MAP VISUALIZATION
    print('Real map start')
    locaz = ((posit.set_index('node')['long'][tot_path[0]]) / 1000000,
             (posit.set_index('node')['lat'][tot_path[0]]) / 1000000)
    m = folium.Map(location=locaz, zoom_start=10,
                   tiles='openstreetmap')

    for node in tot_path:
        folium.CircleMarker(location=((posit.set_index('node')['long'][node]) / 1000000,
                                      (posit.set_index('node')['lat'][node]) / 1000000), radius=3,
                            line_color='#3186cc', fill_color='#FFFFFF', fill_opacity=0.7, fill=True).add_to(m)
    folium.CircleMarker(location=((posit.set_index('node')['long'][tot_path[0]]) / 1000000,
                                  (posit.set_index('node')['lat'][tot_path[0]]) / 1000000), radius=7,
                        line_color='red', fill_color='red', fill_opacity=0.7, fill=True,
                        popup=('Start: ' + str(tot_path[0]))).add_to(m)
    m.save('frontend1.html')
    print('frontend1.html saved')


def func_1_distance(node, th):

    print('Func_1_distance begin')
    s = time.time()
    data = distance_df()
    g = []
    graph = visiting(data, node, th, g)
    graph = pd.DataFrame(graph)
    print(graph)
    graph.columns = ['iniz', 'end', 'weight']
    print('Backend finish whit time:', time.time() - s)

    return graph


def frontend_f1_distance(sol):
    posit = position()
    posi = dict([(i, (a, b)) for i, a, b in zip(posit.node, posit.long, posit.lat)])
    G = nx.from_pandas_edgelist(sol, 'iniz', 'end', 'weight')

    color_map = []
    tot_path = []
    for node in G:
        color_map.append('green')
        tot_path.append(node)
    color_map[0] = 'b'
    nx.draw(G, posi, node_color=color_map, with_labels=True, edge_color='r')

    plt.show()

    # REAL MAP VISUALIZATION
    print('Real map start')
    locaz = ((posit.set_index('node')['long'][tot_path[0]]) / 1000000,
             (posit.set_index('node')['lat'][tot_path[0]]) / 1000000)
    m = folium.Map(location=locaz, zoom_start=10,
                   tiles='openstreetmap')

    for node in tot_path:
        folium.CircleMarker(location=((posit.set_index('node')['long'][node]) / 1000000,
                                      (posit.set_index('node')['lat'][node]) / 1000000), radius=3,
                            line_color='#3186cc', fill_color='#FFFFFF', fill_opacity=0.7, fill=True).add_to(m)
    folium.CircleMarker(location=((posit.set_index('node')['long'][tot_path[0]]) / 1000000,
                                  (posit.set_index('node')['lat'][tot_path[0]]) / 1000000), radius=7,
                        line_color='red', fill_color='red', fill_opacity=0.7, fill=True,
                        popup=('Start: ' + str(tot_path[0]))).add_to(m)
    m.save('frontend1.html')
    print('frontend1.html saved')


def func_1_net(node, th):

    print('Func_1_net begin')
    s = time.time()
    data = net_dist()
    g = []
    graph = visiting(data, node, th, g)
    graph = pd.DataFrame(graph)
    print(graph)
    graph.columns = ['iniz', 'end', 'weight']
    print('Backend finish whit time:', time.time() - s)

    return graph


def frontend_f1_net(sol):
    posit = position()
    posi = dict([(i, (a, b)) for i, a, b in zip(posit.node, posit.long, posit.lat)])
    G = nx.from_pandas_edgelist(sol, 'iniz', 'end', 'weight')

    color_map = []
    tot_path = []
    for node in G:
        color_map.append('r')
        tot_path.append(node)
    color_map[0] = 'b'
    nx.draw(G, posi, node_color=color_map, with_labels=True, edge_color='g')

    plt.show()

    # REAL MAP VISUALIZATION
    print('Real map start')
    locaz = ((posit.set_index('node')['long'][tot_path[0]]) / 1000000,
             (posit.set_index('node')['lat'][tot_path[0]]) / 1000000)
    m = folium.Map(location=locaz, zoom_start=10,
                   tiles='openstreetmap')

    for node in tot_path:
        folium.CircleMarker(location=((posit.set_index('node')['long'][node]) / 1000000,
                                      (posit.set_index('node')['lat'][node]) / 1000000), radius=3,
                            line_color='#3186cc', fill_color='#FFFFFF', fill_opacity=0.7, fill=True).add_to(m)
    folium.CircleMarker(location=((posit.set_index('node')['long'][tot_path[0]]) / 1000000,
                                  (posit.set_index('node')['lat'][tot_path[0]]) / 1000000), radius=7,
                        line_color='red', fill_color='red', fill_opacity=0.7, fill=True,
                        popup=('Start: ' + str(tot_path[0]))).add_to(m)
    m.save('frontend1.html')
    print('frontend1.html saved')

