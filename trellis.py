import itertools
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    def __init__(self, state, col):
        self.state = state
        self.col = col
        self.connections = []  # List of tuples (next_node, info_bit, output_bits, hamming_weight)

    def connect(self, next_node, info_bit, output_bits, hamming_weight):
        self.connections.append((next_node, info_bit, output_bits, hamming_weight))

def calc_output(reg, poly1_powers, poly2_powers):
    output1 = 0
    output2 = 0
    
    for i in range(len(poly1_powers)):
        if poly1_powers[i] == 1:
            output1 ^= reg[i]       
    for i in range(len(poly2_powers)):
        if poly2_powers[i] == 1:
            output2 ^= reg[i]
    return (output1 << 1) | output2

def state_string(num_of_states):
    k = num_of_states.bit_length() - 1  # number of bits per string
    states = [format(i, f'0{k}b') for i in range(num_of_states)]
    states.sort(key=lambda x: (x[0], x[1]))
    return states

def create_trellis(poly1_powers, poly2_powers, input_bits, K, d):
    num_of_states = 2**(K-1)
    states = state_string(num_of_states)
    mapping = {state: idx for idx, state in enumerate(states)}
    cols = input_bits+K
    nodes = [[Node(curr_state, j) for j in range(cols)] for curr_state in states]
        
    for bits in itertools.product([0, 1], repeat=input_bits):
        bits = list(bits) + [0]*(K-1)
        curr_state = "00"
        reg = [0]*K
        for i in range(0, cols-1):
            found = 0
            bit = bits[i]
            reg = [bit] + reg[:-1]
            output = calc_output(reg, poly1_powers, poly2_powers)
            curr_d = ''.join(str(d[j]) for j in range(i*2, i*2+2))
            next_state = ''.join(str(reg[j]) for j in range(2))
            hamming_dist = sum(x!=y for x,y in zip(curr_d, states[output]))
            
            # Get indices for current and next states
            curr_idx = mapping[curr_state]
            next_idx = mapping[next_state]

            for next_node, _, _, _ in nodes[curr_idx][i].connections:
                if next_node.state == next_state:
                    found = 1 
            if not found:    
                nodes[curr_idx][i].connect(nodes[next_idx][i+1], bit, f"{output:02b}", hamming_dist)
            
            #print(f"node_connections: {nodes[curr_idx][i].connections}")
            curr_state = next_state
    
    return nodes

def visualize_trellis(trellis):    
    G = nx.DiGraph()
    
    # Add all nodes with unique IDs but store original state for labels
    for i in range(len(trellis)):
        for col in range(len(trellis[0])):
            node_id = f"{trellis[i][col].state}_{col}"
            G.add_node(node_id, pos=(col, -i), state_label=f"{trellis[i][col].state}")
    
    edge_styles = []  # store edge styles ('solid' or 'dashed')
    edge_colors = []  
    edge_list = []    # actual edges for drawing
    
    # Add edges using the unique IDs
    for i in range(len(trellis)):
        for col in range(len(trellis[0])-1):
            for next_node, bit, output_bits, weight in trellis[i][col].connections:
                source = f"{trellis[i][col].state}_{col}"
                target = f"{next_node.state}_{next_node.col}"
                edge_label = f"{bit}/{output_bits}/{weight}"
                G.add_edge(source, target, label=edge_label)
                edge_list.append((source, target))
                edge_styles.append('dashed' if bit == 1 else 'solid')
                edge_colors.append('black')  # or any other logic

    node_labels = {node: G.nodes[node]['state_label'] for node in G.nodes()}
    pos = nx.get_node_attributes(G, 'pos')

    # Draw nodes with custom color
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700)

    # Draw edges with styles
    for edge, style, color in zip(edge_list, edge_styles, edge_colors):
        if style == 'dashed':
            # For dashed lines, we need to use matplotlib's line style syntax
            nx.draw_networkx_edges(G, pos, edgelist=[edge], 
                                edge_color=color, width=1.0, 
                                style=(0, (3, 5)))  # (0, (dash_length, space_length))
        else:
            nx.draw_networkx_edges(G, pos, edgelist=[edge], style='solid', 
                                edge_color=color, width=1.5)
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    poly1 = [1, 0, 1]  # 1 + x^2
    poly2 = [1, 1, 1]  # 1 + x + x^2
    K = 3 # memory (reg) size
    input_length = 6 
    d = [0,1,1,1,1,0,0,0,1,0,1,0,1,1,0,0]

    trellis = create_trellis(poly1, poly2, input_length, K, d)
    visualize_trellis(trellis)