import re
import spacy
import networkx as nx
import matplotlib.pyplot as plt

class CausalQuestionGraphParser:
    """
    A general and robust parser that converts factual questions into directed causal graphs.
    
    This parser is domain-agnostic and relies on linguistic patterns rather than
    specific keywords to identify and represent causal relationships.
    """
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Define causal indicators and their directions. These are based on
        # common linguistic patterns for expressing causality.
        self.causal_indicators = {
            'temporal': {
                'verbs': ['built', 'constructed', 'created', 'established', 'founded', 'erected', 'formed'],
                'direction': 'cause_to_exist'
            },
            'material': {
                'verbs': ['made', 'composed', 'built', 'consist'],
                'prepositions': ['of', 'from', 'with', 'in'],
                'direction': 'enables_property'
            },
            'functional': {
                'verbs': ['designed', 'intended', 'equipped', 'have', 'run', 'work', 'allows', 'enables'],
                'direction': 'enables_function'
            },
            'locational': {
                'verbs': ['located', 'situated', 'positioned', 'stretch'],
                'prepositions': ['in', 'across', 'through'],
                'direction': 'geographic_influence'
            },
            'quantitative': {
                'indicators': ['over', 'more than', 'spanning', 'stretching', 'due to', 'because of'],
                'direction': 'scale_enables'
            }
        }
    
    def preprocess_question(self, question):
        """Converts an interrogative question into a declarative statement using regex."""
        question = question.strip().rstrip('?').strip()
        patterns = [
            (r'^Is\s+(.+)', r'\1 is'), (r'^Are\s+(.+)', r'\1 are'),
            (r'^Does\s+(.+?)\s+(.*)', r'\1 \2'), (r'^Do\s+(.+)', r'\1'),
            (r'^Can\s+(.+)', r'\1 can'), (r'^Was\s+(.+)', r'\1 was'),
            (r'^Were\s+(.+)', r'\1 were'), (r'^Has\s+(.+)', r'\1 has'),
            (r'^Have\s+(.+)', r'\1 have'), (r'^Why\s+does\s+(.+)', r'\1 does'),
        ]
        for pattern, replacement in patterns:
            new_question = re.sub(pattern, replacement, question, flags=re.IGNORECASE)
            if new_question != question:
                return new_question
        return question
    
    def extract_causal_relationships(self, doc):
        """Extracts causal relationships from a spaCy dependency parse."""
        root_verb = next((token for token in doc if token.dep_ == "ROOT" and token.pos_ == "VERB"), None)
        if not root_verb: return []
        causal_relations = []
        causal_relations.extend(self._extract_temporal_causation(root_verb, doc))
        causal_relations.extend(self._extract_material_causation(root_verb, doc))
        causal_relations.extend(self._extract_locational_causation(root_verb, doc))
        causal_relations.extend(self._extract_quantitative_causation(root_verb, doc))
        causal_relations.extend(self._extract_functional_causation(root_verb, doc))
        return causal_relations
    
    def _extract_temporal_causation(self, verb, doc):
        relations = []
        if verb.lemma_ in self.causal_indicators['temporal']['verbs']:
            subject = self._extract_subject(verb)
            for child in verb.children:
                if child.dep_ == "prep" and child.text in ["during", "in", "by"]:
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            temporal_cause = self._get_complete_phrase(grandchild)
                            if subject and temporal_cause:
                                relations.append({
                                    'cause': f"{temporal_cause} construction",
                                    'effect': f"{subject} existence",
                                    'causal_type': 'temporal_causation',
                                    'strength': 'strong',
                                    'direction': 'historical_to_present'
                                })
        return relations
    
    def _extract_material_causation(self, verb, doc):
        relations = []
        if verb.lemma_ in self.causal_indicators['material']['verbs']:
            subject = self._extract_subject(verb)
            for child in verb.children:
                if child.dep_ == "prep" and child.text in self.causal_indicators['material']['prepositions']:
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            material = self._get_complete_phrase(grandchild)
                            if subject and material:
                                relations.append({
                                    'cause': material,
                                    'effect': f"{subject} structural properties",
                                    'causal_type': 'material_causation',
                                    'strength': 'strong',
                                    'direction': 'material_to_property'
                                })
        return relations
    
    def _extract_locational_causation(self, verb, doc):
        relations = []
        if verb.lemma_ in self.causal_indicators['locational']['verbs']:
            subject = self._extract_subject(verb)
            for child in verb.children:
                if child.dep_ == "prep" and child.text in self.causal_indicators['locational']['prepositions']:
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            location = self._get_complete_phrase(grandchild)
                            if subject and location:
                                # Generalized from "cultural significance"
                                relations.append({
                                    'cause': f"{location} location",
                                    'effect': f"{subject} characteristics",
                                    'causal_type': 'locational_causation',
                                    'strength': 'medium',
                                    'direction': 'location_to_characteristics'
                                })
        return relations
    
    def _extract_quantitative_causation(self, verb, doc):
        relations = []
        for token in doc:
            if token.like_num or token.ent_type_ in ["QUANTITY", "CARDINAL"]:
                measurement_phrase = self._get_complete_phrase(token)
                subject = next((sent_token for sent_token in doc if sent_token.pos_ in ["NOUN", "PROPN"] and any(child.i == token.i for child in sent_token.subtree)), None)
                if not subject: subject = self._extract_subject(verb)
                if subject and measurement_phrase and any(indicator in measurement_phrase.lower() for indicator in ['mile', 'kilometer', 'year', 'meter']):
                    # Generalized from "engineering achievement status"
                    relations.append({
                        'cause': f"{subject} scale",
                        'effect': f"{subject} significance",
                        'causal_type': 'quantitative_causation',
                        'strength': 'medium',
                        'direction': 'scale_to_significance'
                    })
        return relations
    
    def _extract_functional_causation(self, verb, doc):
        relations = []
        if verb.lemma_ in self.causal_indicators['functional']['verbs']:
            subject = self._extract_subject(verb)
            obj = self._extract_direct_object(verb)
            if subject and obj:
                # Generalized from "defensive capability"
                relations.append({
                    'cause': f"{obj} design",
                    'effect': f"{subject} functionality",
                    'causal_type': 'functional_causation',
                    'strength': 'strong',
                    'direction': 'design_to_function'
                })
        return relations
    
    def _extract_subject(self, verb):
        for child in verb.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                return self._get_complete_phrase(child)
        return None
    
    def _extract_direct_object(self, verb):
        for child in verb.children:
            if child.dep_ == "dobj":
                return self._get_complete_phrase(child)
        return None
    
    def _get_complete_phrase(self, token):
        phrase_tokens = list(token.subtree)
        phrase_tokens.sort(key=lambda x: x.i)
        return " ".join([t.text for t in phrase_tokens]).strip()
    
    def build_causal_graph(self, causal_relations):
        """Builds a directed causal graph from a list of relationships."""
        G = nx.DiGraph()
        for rel in causal_relations:
            cause, effect = rel['cause'], rel['effect']
            causal_type, strength = rel['causal_type'], rel['strength']
            G.add_node(cause, node_type='cause')
            G.add_node(effect, node_type='effect')
            G.add_edge(cause, effect, causal_type=causal_type, strength=strength, label=f"{causal_type} ({strength})")
        return G
    
    def question_to_causal_graph(self, question):
        """Main method to convert a question to a directed causal graph."""
        statement = self.preprocess_question(question)
        doc = self.nlp(statement)
        causal_relations = self.extract_causal_relationships(doc)
        graph = self.build_causal_graph(causal_relations)
        return graph, causal_relations
    
    def visualize_causal_graph(self, graph, title="Causal Knowledge Graph"):
        """Visualizes the directed causal graph."""
        plt.figure(figsize=(14, 10))
        try: pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
        except: pos = nx.spring_layout(graph, k=3, iterations=50)
        node_colors = ['lightcoral' if graph.nodes[node].get('node_type', 'unknown') == 'cause' else 'lightblue' for node in graph.nodes()]
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=3000, alpha=0.8)
        edge_colors, edge_styles = [], []
        for u, v, data in graph.edges(data=True):
            causal_type = data.get('causal_type', 'unknown')
            if causal_type == 'temporal_causation': edge_colors.append('red')
            elif causal_type == 'material_causation': edge_colors.append('blue')
            elif causal_type == 'locational_causation': edge_colors.append('green')
            elif causal_type == 'functional_causation': edge_colors.append('purple')
            elif causal_type == 'quantitative_causation': edge_colors.append('orange')
            else: edge_colors.append('gray')
            edge_styles.append('solid')
        nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, style=edge_styles, arrows=True, arrowsize=25, arrowstyle='->', width=2)
        nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='bold')
        edge_labels = nx.get_edge_attributes(graph, 'label')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=6)
        plt.title(title, fontsize=16, fontweight='bold')
        legend_text = """
        Node Colors: Red = Causes, Blue = Effects
        Edge Colors: Red = Temporal, Blue = Material, Green = Locational, Purple = Functional, Orange = Quantitative
        Arrows show causal direction: Cause → Effect
        """
        plt.figtext(0.02, 0.02, legend_text, fontsize=8, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        plt.axis('off')
        plt.tight_layout()
        plt.show()


# Test with 10 factual questions
def main():
    questions = [
        "Is the Great Wall of China located in China?",
        "Does the Great Wall span over 13000 miles?", 
        "Was the Great Wall built during the Ming Dynasty?",
        "Can the Great Wall be seen from space?",
        "Is the Great Wall made of stone and brick?",
        "Does the Great Wall have watchtowers?",
        "Was the Great Wall constructed over 2000 years?",
        "Is the Great Wall a UNESCO World Heritage Site?",
        "Does the Great Wall stretch across northern China?",
        "Are millions of tourists visiting the Great Wall annually?"
    ]
    
    parser = CausalQuestionGraphParser()
    
    # Process all questions and combine into one causal graph
    combined_graph = nx.DiGraph()
    all_causal_relations = []
    
    for question in questions:
        graph, relations = parser.question_to_causal_graph(question)
        
        # Merge into combined graph
        combined_graph = nx.compose(combined_graph, graph)
        all_causal_relations.extend(relations)
        
        print("-" * 70)
    
    print(f"\nTotal causal relationships extracted: {len(all_causal_relations)}")
    print(f"Combined causal graph has {combined_graph.number_of_nodes()} nodes and {combined_graph.number_of_edges()} edges")
    
    # Show sample causal relationships
    print("\nSample Causal Relationships:")
    for i, rel in enumerate(all_causal_relations[:5]):
        print(f"{i+1}. {rel['cause']} → {rel['causal_type']} → {rel['effect']}")
    
    # Visualize the combined causal graph
    if combined_graph.number_of_nodes() > 0:
        parser.visualize_causal_graph(combined_graph, "Great Wall Causal Knowledge Graph")
    else:
        print("No causal relationships found to visualize.")
    
    return combined_graph, all_causal_relations

if __name__ == "__main__":
    graph, relations = main()
