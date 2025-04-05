import itertools
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    def __init__(self, state, col):
        self.state = state
        self.col = col
        self.connections = []  # List of tuples (next_node, info_bit)

    def connect(self, next_node, info_bit):
        self.connections.append((next_node, info_bit))

def calc_state(reg, poly1_powers, poly2_powers):
    output1 = 0
    output2 = 0
    
    for pos in poly1_powers:
        output1 ^= reg[pos]       
    for pos in poly2_powers:
        output2 ^= reg[pos]

    return (output2 << 1) | output1

def create_trellis(poly1_powers, poly2_powers, n):
    states = ["00", "10", "01", "11"] 
    nodes = [[Node(curr_state, j) for j in range(n+1)] for curr_state in states]
    for reg in itertools.product([0, 1], repeat=(n-2)):
        reg = list(reg) + [0]*(n) 
        curr_state = 0
        new_reg = [0]*n
        for i in range (0,n):
            bit = reg[i]
            new_reg = [bit] + new_reg[:-1]
            print(f"reg:     {reg}")
            print(f"new_reg: {new_reg}")
            next_state = calc_state(new_reg, poly1_powers, poly2_powers)
            nodes[curr_state][i].connect(nodes[next_state][i+1],bit)
            curr_state = next_state
    return nodes

def visualize_trellis(trellis):    
    G = nx.DiGraph()
    
    # Add all nodes with unique IDs but store original state for labels
    for i in range(len(trellis)):
        for col in range(len(trellis[0])):
            # Create unique ID internally
            node_id = f"{trellis[i][col].state}_{col}"
            # Store the state as a separate attribute for labeling
            G.add_node(node_id, pos=(col, -i), state_label=f"{trellis[i][col].state}")
    
    # Add edges using the unique IDs
    for i in range(len(trellis)):
        for col in range(len(trellis[0])-1):
            for next_node, bit in trellis[i][col].connections:
                source = f"{trellis[i][col].state}_{col}"
                target = f"{next_node.state}_{next_node.col}"
                G.add_edge(source, target, label=str(bit))
    
    # Create a dictionary mapping nodes to their labels (just the state)
    node_labels = {node: G.nodes[node]['state_label'] for node in G.nodes()}
    
    # Draw with custom labels
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, labels=node_labels, node_size=700)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

poly1_powers = [1, 2, 4, 5, 6]  
poly2_powers = [0, 1, 6]
n = 8
visualize_trellis(create_trellis(poly1_powers,poly2_powers,n))


