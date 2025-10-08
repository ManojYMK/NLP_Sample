# social_network_diffusion_demo.py
# Demo of diffusion on a small directed social graph (Independent Cascade)
# Author: Manoj Kumar Yasangi

import random
import networkx as nx
import matplotlib.pyplot as plt

def build_graph():
    G = nx.DiGraph()
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("C", "E"), ("D", "E")]
    G.add_edges_from(edges)
    return G

def simulate_icm(G, seed, activation_prob=0.35):
    """Simplified Independent Cascade simulation."""
    active = {seed}
    frontier = {seed}
    steps = [set(frontier)]
    while frontier:
        next_frontier = set()
        for u in frontier:
            for v in G.successors(u):
                if v not in active and random.random() < activation_prob:
                    next_frontier.add(v)
        if not next_frontier:
            break
        steps.append(set(next_frontier))
        active |= next_frontier
        frontier = next_frontier
    return active, steps

if __name__ == "__main__":
    G = build_graph()
    seed = "A"
    activated, steps = simulate_icm(G, seed, activation_prob=0.35)
    print(f"Seed: {seed}")
    print(f"Activated nodes: {sorted(activated)}")
    print(f"Steps: {steps}")

    pos = nx.spring_layout(G, seed=7)
    colors = ["lightgreen" if n in activated else "lightgray" for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=800, arrows=True)
    plt.title("Independent Cascade on a Toy Social Graph")
    plt.show()

