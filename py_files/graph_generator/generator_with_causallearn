import spacy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq

class CausalLearnGraphParser:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

            
    def extract_entities(self, question):
        """Extract simple noun phrases from the question."""
        doc = self.nlp(question)
        entities = [chunk.text.strip() for chunk in doc.noun_chunks]
        return list(set(entities))  # unique entities

    
    def build_dummy_data(self, entities_list):
        """
        Build synthetic binary dataset where each question = row,
        each entity = column (1 if entity appears, else 0).
        """
        all_entities = sorted(set(e for ents in entities_list for e in ents))
        data = np.zeros((len(entities_list), len(all_entities)))
        for i, ents in enumerate(entities_list):
            for e in ents:
                data[i, all_entities.index(e)] = 1
        return data, all_entities

    
    def run_causal_discovery(self, data, var_names):
        """Run PC algorithm with proper variable naming."""
        # Create graph with our variable names
        cg = pc(data, alpha=0.05, indep_test=chisq, show_progress=False)

        # Properly set node names
        for i, node in enumerate(cg.G.nodes):
            node.set_name(var_names[i])

        return cg

    
    def visualize_graph(self, cg, title="Causal Graph"):
        """Robust visualization without layout warnings."""
        G = nx.DiGraph()

        # Get nodes and names
        nodes = cg.G.nodes
        node_names = [node.get_name() for node in nodes]

        # Add nodes
        G.add_nodes_from(node_names)

        # Add edges from adjacency matrix
        adj_matrix = cg.G.graph
        for i, j in zip(*np.where(adj_matrix != 0)):
            if adj_matrix[i,j] == 1:  # Directed edge
                G.add_edge(node_names[i], node_names[j])
            elif adj_matrix[i,j] == -1:  # Undirected edge
                G.add_edge(node_names[i], node_names[j])
                G.add_edge(node_names[j], node_names[i])

        # Create figure with constrained layout instead of tight_layout
        fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')

        # Choose layout based on node count
        pos = nx.circular_layout(G) if len(G) <= 5 else nx.spring_layout(G, k=1.5, seed=42)

        # Draw with enhanced visibility
        nx.draw_networkx(
            G, pos, ax=ax,
            node_size=2000,
            node_color='lightblue',
            font_size=12,
            font_weight='bold',
            arrowsize=25,
            width=2,
            edge_color='black',
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.1'
        )

        # Add title and handle empty graphs
        ax.set_title(title, fontsize=16, pad=20)
        if len(G.edges()) == 0:
            ax.text(0.5, 0.5, "No causal relationships found", 
                   ha='center', va='center', fontsize=14)

        plt.show()


# === Example usage ===
if __name__ == "__main__":
    questions = [
        "Do heatwaves increase heat-related illness in cities?",
        "Is heat-related illness more common during heatwaves?", 
        "Can heatwaves trigger spikes in heat-related illness",
        "Are heat-related illness incidents higher in prolonged heatwaves",
        "Do multi-day heatwaves worsen heat-related illness outcomes?",
    ]

    parser = CausalLearnGraphParser()

    # Step 1: Extract entities
    entities_list = [parser.extract_entities(q) for q in questions]

    # Step 2: Build dummy binary data
    data, var_names = parser.build_dummy_data(entities_list)
    print("Variable names:", var_names)  # Verify names are correct

    # Step 3: Run causal discovery
    cg = parser.run_causal_discovery(data, var_names)

    # Step 4: Visualize
    parser.visualize_graph(cg, title="Causal Graph from Questions")
