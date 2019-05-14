# Project Title: In All Fairness 
# Network Science Final Project
#
# Author Shota Nakamura
# Date: 5/14/2019
#  --------------------------------------------------------
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import collections

"""
Let nodes represent players and let there be an edge between two players
represent a game to be played between them.

In this project, I will analyze the "fairness" of tournament brackets.

I define Fairness as the following:
Fairness = small sensitivity of initial conditions (Low variance)


"""

def deterministicMat(num_nodes):
    """
    num_nodes: Number of nodes in graph
    returns: List of Lists s.t. A[m][n] = 1 if m will always beat [n]
             (Creates Deterministic Matrix)
    """
    return [random.random() for i in range(0,num_nodes)]


def probabilisticMat(num_nodes):
    """
    num_nodes: Number of nodes in graph
    returns: List of Lists s.t. A[m][n] = prob. of m to beat n
             (Creates Probabilistic Matrix)
             if A[m][n] = k, A[n][m] = 1-k
    """
    # initialize empty matrix num_nodesxnum_nodes
    A = [[0 for i in range(0,num_nodes)] for j in range(0,num_nodes)]

    # List to check if random num already has been generated
    genList = []
    i = 0
    while i != num_nodes:
        j = 0
        while j != num_nodes:
            if(A[i][j] == 0):
                gen = random.random()
                while gen in genList:
                    gen = random.random()
                if gen not in genList:
                    A[i][j] = gen
                    A[j][i] = 1 - A[i][j]

            j += 1
        i += 1
    return A

def initializePlayers(num_nodes):
    """
    num_nodes: Number of players/nodes in bracket
    returns: Graph with num_nodes nodes and no edges
    """
    bracket = [i for i in range(0,num_nodes)]
    G = nx.Graph()
    for i in range(0,num_nodes):
        G.add_node(i)
    return G

def initMatchups(G,randomize = True):
    """
    G: Graph with n players without edges
    returns: Graph with edges of initial matchups

    """
    nodes = list(G.nodes())

    # Shuffle nodes in graph and split them into two groups
    if randomize == True:
        random.shuffle(nodes)
    region1 = nodes[:(len(nodes)//2)]
    region2 = nodes[(len(nodes)//2):]

    while len(region1) > 0 and len(region2) > 0:
        i = region1.pop()
        j = region2.pop()
        G.add_edge(i,j)
    
    return G

def playMatches(G,matrix,prob,ranking):
    """
    G: graph with edges representing games
    returns: Tuple of
                G: Resulting state of bracket
                remPlayers: Remaining players of bracket
    """
    edges = list(G.edges())
    remPlayers = []
    num_nodes = len(G.nodes())
    for edge in edges:
        m = edge[0]
        n = edge[1]
        if prob:
            if random.random() < matrix[m][n]:
                remPlayers.append(m)
                ranking[m] += 1
            else:
                remPlayers.append(n)
                ranking[n] += 1
        else: 
            if matrix[m] > matrix[n]:
                remPlayers.append(m)
                ranking[m] +=1
            else:
                remPlayers.append(n)
                ranking[n] += 1
        G.remove_edge(m,n)
    return (G,remPlayers,ranking)

def listToBracket(G,nextMatches):
    """
    G: graph with no edges
    nextMatches: Takes this list and adds each pair as edge to G
    returns: Graph with edges representing next matches
    """
    while nextMatches != []:
        i = nextMatches.pop()
        j = nextMatches.pop()
        G.add_edge(i,j)
    return G
    
def playAllMatches(G,nextMatches,matrix,prob,ranking):
    """
    G: Graph of Bracket with first round of games already played
    nextMatches: next set of matches to be played
    matrix: used to determine which node wins
    prob: True if matrix is probabilistic
    returns: ranking of each node
    """
    while len(nextMatches) != 1:
        # Do not randomize order.
        G = listToBracket(G,nextMatches)
        (G,nextMatches,ranking) = playMatches(G,matrix,prob,ranking)    
    
    # We will return dict of ranking of nodes here
    return ranking


# Now we must do above simulation num_simulation times.
def runSimulation(num_sim,num_nodes,dM,pM,flag):
    """
    num_sim: Number of Simulations
    num_nodes: Number of nodes in graph
    dM: Deterministic Matrix
    pM: Probablistic Matrix
    flag: True if probabilistic
    returns: dict of rankings
    """
    listRanks = {i:[] for i in range(0,num_nodes)}
    for i in range(0,num_sim):
        # Add nodes to graph
        g = initializePlayers(num_nodes)
        # Create initial matchups
        c = initMatchups(g)
        # initialize rankings
        init_ranking = {i:0 for i in range(0,num_nodes)}
        # play first match 
        if flag:
            flagged = dM
        else:
            flagged = pM
        (afterFirstMatch,nextMatches,ranking) = playMatches(c,flagged,False,init_ranking)
        # play rest of matches
        final = playAllMatches(afterFirstMatch,nextMatches,flagged,False,ranking)
        
        for key,val in final.items():
            listRanks[key].append(val)
    return listRanks
        
        
"""
Test Code
"""
num_nodes = 32
# Initialize Matrices
dM = deterministicMat(num_nodes)
pM = probabilisticMat(num_nodes)

# # Simulation for dM
allRanks = runSimulation(10000,num_nodes,dM,pM,True)

varValues = {}
for player,rankList in allRanks.items():
    varValues[player] = np.var(rankList)
print(varValues)

# Number of Simulations vs Final Rank
x = [i for i in range(10000)]
for i in range(0,num_nodes):
    y = allRanks[i]
    plt.hist(y)
    plt.xlabel('Number of Simulations')
    plt.ylabel('Rank')
    plt.show()

# simulation for pM
allRanks2 = runSimulation(10000,num_nodes,dM,pM,True)
varValues = {}
for player,rankList in allRanks2.items():
    varValues[player] = np.var(rankList)

# Number of Simulations vs Final Rank
x = [i for i in range(10000)]
for i in range(0,num_nodes):
    y = allRanks2[i]
    plt.hist(y, bins=4)
    plt.xlabel('Number of Simulations')
    plt.ylabel('Rank')
    plt.show()

"""
The variance does not give us a good measure of the probability
distribution since this is not quantitative data. What we have 
here is categorical data and entropy works particularly well for
categorical data. Variance is actually sensitive to this kind of data
where we can change the scale accordingly.

Therefore, instead we use the entropy of the 
probability distribution to find a better measure for fairness.
"""
def computeEntropy(rankingList,num_nodes):
    result = {i:0 for i in range(0,num_nodes)}
    for i in range(0,num_nodes):
        one = rankingList[i]
        d = collections.Counter(one)
        v = np.array(sorted(d.values()),dtype=float)
        v /= v.sum()
        H = -(v*np.log(v)).sum()
        result[i] = H
    return result

# Entropy dict of deterministic
dM_entrop = computeEntropy(allRanks,num_nodes)

# Entropy dict of probabilistic
pM_entrop = computeEntropy(allRanks2,num_nodes)

"""
For a tournament bracket of size 32, the worst we can do
is entropy = log 32 which is approximately 1.50514997832
"""
