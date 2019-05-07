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


"""
Let nodes represent players and let there be an edge between two players
represent a game to be played between them.

In this project, I will analyze the "fairness" of tournament brackets.

I define Fairness as the following:
Fairness = small sensitivity of initial conditions (Low variance)


"""
def runCreateBracket(num_nodes, init = True,G=nx.Graph(),ordered=False,rand=False):
    """
    num_nodes: Number of nodes
    init: False you have a non empty graph
    ordered: True if you want to sort nodes by highest win rate
    rand: True if you want to randomize next round of games every round
    returns: Output of createBracket which is a networkx graph representing
             a tournament bracket
    """
    if ordered == True:
        randVals = [random.random() for i in range(0,num_nodes)]
        randVals = list(sorted(randVals))
        winProb = {i:randVals[i] for i in range(0,num_nodes)}
    else:
        winProb = winProb = {i:random.random() for i in range(0,num_nodes)}
    # Here we need to create a new dictionary to keep track of variance
    varTracker = {var:0 for var in range(0,num_nodes)}
    return createBracket(num_nodes,winProb,init,G,ordered,rand)

def createBracket(num_nodes, winProb,init = True, G=nx.Graph(), ordered=False,rand = False):
    """
    num_nodes: Number of nodes in Bracket
    init: True if initial Graph is empty
    ordered: True if assigned win probability is in ascending order
    returns: A Bracket with matches in next round as edges
    """
    if(init):
        # Separate into two regions
        region1 = [i for i in range(0,num_nodes // 2)]
        region2 = [i for i in range(8,num_nodes)]
    else:
        nodes = list(G.nodes())
        region1 = nodes[:len(nodes)//2]
        region2 = nodes[len(nodes)//2:]

    # Suppose the initial bracket is made so each player in region1
    # and region2 play each other
    if(rand == True):
        random.shuffle(region1)
        random.shuffle(region2)
    while (len(region1) > 0 and len(region2) > 0):
        i = region1.pop()
        j = region2.pop()
        G.add_edge(i,j)

    nx.set_node_attributes(G, winProb, 'winProb')
    
    return G

def playBracket(G,varTracker, winList = [],prob=False,game_num = 0,extraEdges = 0):
    """
    G: Network x graph that represents a bracket to be played containing
        edges that represent the first round of matches
    winList: Matrix m*n that represents chances of m beating n
    prob: True if matrix has probability instead of 1 or 0
    game_num: Number of rounds played so far
    extraEdges: Number of extra edges to randomly add in prelim.
    returns: graph with no edges s.t. nodes remaining in graph
             won the previous round of matches
    """
    edgeList = list(G.edges())

    while len(edgeList) > 0:
        game = edgeList.pop()
        node1 = game[0]
        node2 = game[1]
        G.remove_edge(node1,node2)
        if(winList == []):
            if(G.nodes[node1]['winProb'] >= G.nodes[node2]['winProb']):
                G.remove_node(node2)
                varTracker[node2] = varTracker[node2] + game_num
            else:
                G.remove_node(node1)
                varTracker[node1] = varTracker[node1] + game_num
        else:
            if(prob == False):
                if(winList[node1][node2] == 1):
                    G.remove_node(node2)
                    varTracker[node2] = varTracker[node2] + game_num
                else:
                    G.remove_node(node1)
                    varTracker[node1] = varTracker[node1] + game_num
            else:
                if(random.random() < winList[node1][node2]):
                    G.remove_node(node2)
                    varTracker[node2] = varTracker[node2] + game_num

                else:
                    G.remove_node(node1)
                    varTracker[node1] = varTracker[node1] + game_num
    return (G,varTracker)


def playTournament(varTrack,bracket=runCreateBracket(100),winList = [],prob = False, rand = False):
    """
    bracket: networkx Graph G representing a bracket
    returns: networkX graph G with one node that won the tournament
    """
    game_num = 0
    # ----------Debug Statements ------------------
    # print("Initial Game State: \n")
    # print("List of Players: ",list(bracket.nodes()))
    # print("List of Matches: ",list(bracket.edges()))
    # --------------------------------------------------
    while len(bracket.nodes()) > 1:
        game_num +=1
        (bracket,varTracker) = playBracket(bracket,varTrack,winList = winList,prob=prob,game_num=game_num)
        bracket = createBracket(num_nodes = len(bracket.nodes()), rand=rand,winProb = nx.get_node_attributes(bracket, 'winProb'),G = bracket,init=False)
        # ----------Debug Statements ------------------
        # print("After Game 1:")
        # print("List of Players: ",list(bracket.nodes()))
        # print("List of Matches: ",list(bracket.edges()))
        # print("Variance Tracker: ",varTrack)
        # ------------------------------------------------
    return varTrack

# What if we initialize a matrix that determines whether a 
# player will win against another?

def makeMatrix(num_nodes):
    """
    num_nodes: Number of nodes in graph
    returns: List of Lists s.t. A[m][n] = 1 if m will always beat [n]
    """
    return [[random.randint(0,1) for i in range(0,num_nodes)] for j in range(0,num_nodes)] 

def makeMatrixProb(num_nodes):
    """
    num_nodes: Number of nodes in graph
    returns: List of Lists s.t. A[m][n] = prob. of m to beat n
    """
    return [[random.random() for i in range(0,num_nodes)] for j in range(0,num_nodes)] 


def calculateVariance(num_nodes,winList=[],prob=False,overall=True):
    """
    num_nodes: Number of Players
    winList: win probability
    prob: True if probabilistic model
    overall: True if you want to calculate overall variance
    """
    variances = []
    for i in range(0,10):
        varTracker = {j:0 for j in range(0,num_nodes)}
        runFirst = playTournament(varTracker,bracket= runCreateBracket(num_nodes),winList=winList,prob=prob)
        for i in range(0,100):
            k = playTournament(runFirst,bracket=runCreateBracket(num_nodes),winList = winList,prob=prob)
            playTournament(k,bracket=runCreateBracket(num_nodes),winList=[],prob=prob)
            variances.append(k)
    for key, value in k.items(): 
        k[key] = value/10

    if overall == True:
        calcVar= 0
        while variances != []:
            var = variances.pop()
            for key,value in var.items():
                # subtract mean from each var(i)
                calcVar += pow((value/10) - k[key],2)
        variance = {"overall": (calcVar/((num_nodes-1)*100))/1000}
    else:
        variance = {i:0 for i in range(0,num_nodes)}
        while variances != []:
            var = variances.pop()
            for key,value in var.items():
                indVar = (pow((value/10) - k[key],2))/((num_nodes-1)*100)/1000
                variance[key] += indVar 
    return variance


# Next we calculate variance for each
# -------------------Constants -----------------------
# Initialize number of nodes in graph
num_nodes = 100  
# # --------------------Play Original Tournament-------------------------
# print("\n\nPlay Original Coin Flip Random Win rate Assignment Tournament")
# print("Overall Fairness: ",calculateVariance(num_nodes = num_nodes))
# print("Individual Fairness: ",calculateVariance(num_nodes = num_nodes,overall=False))

# ----------Deterministic Tournament------------
print("\n\nPlay Deterministic Tournament")
print("Overall Fairness: ",calculateVariance(num_nodes = num_nodes,winList=makeMatrix(num_nodes)))
print("Individual Fairness: ",calculateVariance(num_nodes = num_nodes,winList=makeMatrix(num_nodes),overall=False))


# ----------Probabilistic Tournament-------------------
print("\n\nPlay Probabilistic Tournament")
mat2 = makeMatrixProb(num_nodes)
print("Overall Fairness: ",calculateVariance(num_nodes = num_nodes,winList=makeMatrixProb(num_nodes),prob=True))
print("Individual Fairness: ",calculateVariance(num_nodes = num_nodes,winList=makeMatrixProb(num_nodes),prob=True,overall=False))


"""
Here we ask the question, if we are allowed to add one edge, 
where should we add it to increase fairness?
"""






# If you make a random assignment
# Analyze variance ->

# To do the above, we create a dictionary to keep track of ranking.



"""
As you increase number of preliminary rounds game will become more fair
"""

"""
each person plays 5 random match ups, people who win 3 times go to next round.
"""

"""
If we have time for one more game where should we throw the edge?
"""



"""
Should we have an edge between losers?
"""

"""
System to find observables.
-----------------------------
variance for best/worst player (highest/worst player)
# of players that were ever ranked high

Given these observables,

y = fairness
x = number of games added at random
"""
k = runCreateBracket(100)
varTracker = {j:0 for j in range(0,num_nodes)}
(g1,varada) = playBracket(k,varTracker,extraEdges=0)
print(g1.nodes())