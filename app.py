import streamlit as st
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Graph, Use Case: Emergency Supply Robot (same graph as Task 3)
# ---------------------------------------------------------------
locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6)
}

hospital_graph = {
    "Pharmacy": {"Main_Corridor": 2.2, "Patient_Wing": 4.1},
    "Main_Corridor": {"Nursing_Station": 2.2},
    "Patient_Wing": {"Laboratory": 5.0},
    "Nursing_Station": {"Laboratory": 3.2, "Emergency_Ward": 6.0},
    "Laboratory": {"Emergency_Ward": 3.2},
    "Emergency_Ward": {}
}


# Heuristic: Euclidean distance
def heuristic(current, goal):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Path reconstruction
def reconstruct_path(came_from, current):
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_cost(path):
    return sum(hospital_graph[a][b] for a, b in zip(path, path[1:]))


# GBFS: f(n) = h(n)
def gbfs(start, goal):
    counter = 0
    frontier = [(heuristic(start, goal), counter, start)]
    came_from = {start: None}
    visited = set()

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, path_cost(path)

        for neighbor in hospital_graph[current]:
            if neighbor not in visited and neighbor not in came_from:
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(frontier, (heuristic(neighbor, goal), counter, neighbor))

    return None, None


# A*: f(n) = g(n) + h(n)
def a_star(start, goal):
    counter = 0
    frontier = [(heuristic(start, goal), counter, start)]
    came_from = {start: None}
    g_cost = {start: 0}
    closed = set()

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            return reconstruct_path(came_from, current), g_cost[current]

        for neighbor, cost in hospital_graph[current].items():
            new_g = g_cost[current] + cost
            if new_g < g_cost.get(neighbor, float("inf")):
                g_cost[neighbor] = new_g
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(frontier, (new_g + heuristic(neighbor, goal), counter, neighbor))

    return None, None


##########################################
# Streamlit GUI Code

st.set_page_config(page_title="Hospital Robot Search Visualizer", page_icon="🤖", layout="centered")

st.title("🏥 Emergency Supply Robot: Search Visualizer")
st.write(
    "Pick a start node, a goal node and a search algorithm (**GBFS** or **A\\***). "
    "The app runs the search on the hospital corridor graph and highlights the solution path."
)

nodes = list(hospital_graph.keys())

start = st.selectbox("Select Initial Node", nodes, index=nodes.index("Pharmacy"))
goal = st.selectbox("Select Goal Node", nodes, index=nodes.index("Emergency_Ward"))
algorithm = st.selectbox("Select Search Algorithm", ["GBFS", "A*"])

if st.button("Run Search"):

    if algorithm == "GBFS":
        path, cost = gbfs(start, goal)
    else:
        path, cost = a_star(start, goal)

    if path is None:
        st.error(f"No path found from {start} to {goal}. (The corridors are one-way, so try a different pair.)")

    else:
        # Display result
        st.subheader("Search Result")
        st.write(f"Algorithm: {algorithm}")
        st.write(f"Solution Path: {' → '.join(path)}")
        st.write(f"Total Path Cost: {cost:.2f}")

        # Visualize NetworkX graph
        G = nx.DiGraph()
        for node, neighbors in hospital_graph.items():
            G.add_node(node)
            for neighbor, weight in neighbors.items():
                G.add_edge(node, neighbor, weight=weight)

        pos = locations
        path_edges = list(zip(path, path[1:]))

        fig, ax = plt.subplots(figsize=(10, 6))

        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=2600, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="lightgreen", node_size=2600, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, node_size=2600, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=3.5,
                               arrows=True, arrowsize=25, node_size=2600, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "weight"),
                                     font_size=9, ax=ax)

        ax.set_title(f"{algorithm} Solution Path")
        ax.axis("off")

        st.pyplot(fig)