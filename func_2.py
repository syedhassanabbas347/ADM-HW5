#!/usr/bin/env python
# coding: utf-8

# ## Homework 5 - Explore California and Nevada with graphs
# 

# In[1]:


import pandas as pd
import networkx as nx 
import os 


# #### In the following cell, we are going to define some functions that read the datasets, and converts them into Pandas DataFrames

# In[2]:


def position():
    position = pd.read_csv(r'position.co', header=None, sep=' ')
    position = position.drop([0], axis=1)
    position.columns = ['node', 'long', 'lat']
    return position


def distance_df():
    distance = pd.read_csv(r'distance.gr', header = None, sep = ' ')
    distance = distance.drop([0], axis=1)
    distance.columns = ['from', 'to', 'weight']
    return distance


def time_df():
    time_ = pd.read_csv(r'time.gr', header = None, sep = ' ')
    time_ = time_.drop([0], axis=1)
    time_.columns = ['iniz', 'end', 'weight']
    return time_


def net_dist():
    net_dist = pd.read_csv(r'time.gr', header = None, sep = ' ')
    net_dist = net_dist.drop([0, 3], axis=1)
    n = [1]*len(net_dist)
    net_dist['weight'] = n
    net_dist.columns = ['from', 'to', 'weight']

    return net_dist


# In[3]:


#Assigning the functions to variables

position = position()
distance = distance_df()
time = time_df()
net_dist = net_dist()


# In[4]:


type(position)


# In[8]:


distance.head()


# In[6]:


distance.head()


# 
#  ### <i> Functionality 2 - Find the smartest Network! </i>
# 
#  It takes in input:
#  
#  - a set of nodes _v = {v\_1, ..., v\_n}_
#  - One of the following distances function: **t(x,y)**, **d(x,y)** or **network distance** (i.e. consider all edges to have weight equal to 1).
# 
# Implement an algorithm that returns the set of roads (edges) that enable the user to visit all the places. We want this set to be the ones whose sum of distances is minimum.
# 
# As a dummy example, a set of input could be {Colosseo, Piazza Venezia, Piazza del Popolo} and therefore the associated set of streets will be {Via dei Fori Imperiali, Via del Corso}.
# 
# 

# In[9]:


from collections import defaultdict 
  
# This class represents a directed graph 
# using adjacency list representation 
class Graph: 
  
    # Constructor 
    def __init__(self): 
  
        # default dictionary to store graph 
        self.graph = defaultdict(list) 
  
    # function to add an edge to graph 
    def addEdge(self,u,v): 
        self.graph[u].append(v) 
  
    # Function to print a BFS of graph 
    def BFS(self, s): 
  
        # Mark all the vertices as not visited 
        visited = [False] * (len(self.graph)) 
  
        # Create a queue for BFS 
        queue = [] 
  
        # Mark the source node as  
        # visited and enqueue it 
        queue.append(s) 
        visited[s] = True
  
        while queue: 
  
            # Dequeue a vertex from  
            # queue and print it 
            s = queue.pop(0) 
            print (s, end = " ") 
  
            # Get all adjacent vertices of the 
            # dequeued vertex s. If a adjacent 
            # has not been visited, then mark it 
            # visited and enqueue it 
            for i in self.graph[s]: 
                if visited[i] == False: 
                    queue.append(i) 
                    visited[i] = True


# In[10]:


g = Graph()


# In[1]:


class Vertex:
    def __init__(self, n):
		self.name = n
        self.neighbors = list()
		
		self.discovery = 0
		self.finish = 0
		self.color = 'black'
	
	def add_neighbor(self, v):
		if v not in self.neighbors:
			self.neighbors.append(v)
			self.neighbors.sort()

class Graph:
	vertices = {}
	time = 0
	
	def add_vertex(self, vertex):
		if isinstance(vertex, Vertex) and vertex.name not in self.vertices:
			self.vertices[vertex.name] = vertex
			return True
		else:
			return False
	
	def add_edge(self, u, v):
		if u in self.vertices and v in self.vertices:
			for key, value in self.vertices.items():
				if key == u:
					value.add_neighbor(v)
				if key == v:
					value.add_neighbor(u)
			return True
		else:
			return False
			
	def print_graph(self):
		for key in sorted(list(self.vertices.keys())):
			print(key + str(self.vertices[key].neighbors) + "  " + str(self.vertices[key].discovery) + "/" + str(self.vertices[key].finish))

	def _dfs(self, vertex):
		global time
		vertex.color = 'red'
		vertex.discovery = time
        time += 1
        for v in vertex.neighbors:
            if self.vertices[v].color == 'black':
                self._dfs(self.vertices[v])
		vertex.color = 'blue'
		vertex.finish = time
		time += 1
		
	def dfs(self, vertex):
		global time
		time = 1
        self._dfs(vertex)

g = Graph()
# print(str(len(g.vertices)))
a = Vertex('A')
g.add_vertex(a)
g.add_vertex(Vertex('B'))
for i in range(ord('A'), ord('K')):
	g.add_vertex(Vertex(chr(i)))

edges = ['AB', 'AE', 'BF', 'CG', 'DE', 'DH', 'EH', 'FG', 'FI', 'FJ', 'GJ', 'HI']
for edge in edges:
    g.add_edge(edge[:1], edge[1:])

g.dfs(a)
g.print_graph()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




