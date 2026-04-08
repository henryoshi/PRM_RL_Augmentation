import numpy as np
import networkx as nx
from PIL import Image
import matplotlib.pyplot as plt


# Input: Occupancy map M, desired number of sample points N, maximum local search radius dmax.
# Output: A graph G = (V, E) using NetworkX, where each node is a (row, col) tuple.
def ConstructPRM(M, N, dmax):
    G = nx.Graph()
    for k in range(N):
        vnew = RandomSample(M)
        AddVertex(G, M, vnew, dmax)
    return G


def AddVertex(G, M, vnew, dmax):
    G.add_node(vnew, pos=vnew)
    for v in list(G.nodes):
        if v != vnew and d(v, vnew) < dmax:
            if AttemptPlan(M, v, vnew):
                G.add_edge(v, vnew, weight=d(v, vnew))


def AttemptPlan(M, v1, v2):
    # Sample points along the straight line at sub-pixel resolution to catch all cells
    num_steps = int(d(v1, v2) * 2) + 1
    for i in range(num_steps + 1):
        t = i / num_steps
        r = int(round(v1[0] + t * (v2[0] - v1[0])))
        c = int(round(v1[1] + t * (v2[1] - v1[1])))
        if M[r][c] == 0:
            return False
    return True


def RandomSample(M):
    # Rejection sampling: draw uniform random cells until a free one is found
    rows, cols = M.shape
    while True:
        r = np.random.randint(0, rows)
        c = np.random.randint(0, cols)
        if M[r, c] > 0:
            return (r, c)


def d(v1, v2):
    return np.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)

def find(M, s, g, G, dmax):
    # Connect s and g into the graph as regular vertices
    AddVertex(G, M, s, dmax)
    AddVertex(G, M, g, dmax)

    n = 100  # samples to add per retry
    while True:
        try:
            path = nx.astar_path(G, s, g, heuristic=d, weight='weight')
            dist = nx.astar_path_length(G, s, g, heuristic=d, weight='weight')
            return path, dist
        except nx.NetworkXNoPath:
            # No path found — add more samples and retry
            for _ in range(n):
                AddVertex(G, M, RandomSample(M), dmax)

if __name__ == '__main__':
    occupancy_map_img = Image.open('occupancy_map.png')
    occupancy_grid = (np.asarray(occupancy_map_img) > 0).astype(int)

    N = 2500   # number of sample points
    dmax = 75  # maximum connection radius in pixels
    s = (635, 140)
    g = (350, 400)

    G = ConstructPRM(occupancy_grid, N, dmax)
    print(f"PRM built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Plot the PRM overlaid on the occupancy map
    fig, ax = plt.subplots()
    ax.imshow(occupancy_map_img, cmap='gray')

    # NetworkX expects pos as {node: (x, y)} where x=col, y=row
    pos = {node: (node[1], node[0]) for node in G.nodes}
    nx.draw_networkx(G, pos=pos, ax=ax,
                     node_size=5, node_color='red',
                     edge_color='blue', width=0.5,
                     with_labels=False)
    plt.show()

    path, distance = find(occupancy_grid, s, g, G, dmax)
    print(f"Path length: {distance:.2f}, nodes in path: {len(path)}")

    # Plot just the path overlaid on the occupancy map (no PRM nodes/edges)
    fig2, ax2 = plt.subplots()
    ax2.imshow(occupancy_map_img, cmap='gray')

    path_rows = [p[0] for p in path]
    path_cols = [p[1] for p in path]
    ax2.plot(path_cols, path_rows, color='red', linewidth=1.5)
    ax2.plot(s[1], s[0], 'go', markersize=8)  # start (green)
    ax2.plot(g[1], g[0], 'bo', markersize=8)  # goal (blue)
    plt.show()


