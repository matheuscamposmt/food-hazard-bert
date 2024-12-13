from graphviz import Digraph

def visualize_hierarchical_classifier(model):
    """
    Visualizes the HierarchicalClassifier architecture.
    """
    dot = Digraph(comment="Hierarchical Classifier", format='png')
    dot.attr(rankdir='LR')  # Left to Right flow

    # Backbone
    dot.node("Backbone", "DeBERTa Backbone")

    # Additional Features
    dot.node("Features", "Additional Features")
    dot.edge("Features", "Backbone")

    # ST1 Heads
    dot.node("ST1_Hazard", "ST1 Hazard Category Head")
    dot.node("ST1_Product", "ST1 Product Category Head")
    dot.edge("Backbone", "ST1_Hazard")
    dot.edge("Backbone", "ST1_Product")

    # ST2 Heads
    dot.node("ST2_Hazard", "ST2 Hazard Head")
    dot.node("ST2_Product", "ST2 Product Head")
    dot.edge("ST1_Hazard", "ST2_Hazard")
    dot.edge("ST1_Product", "ST2_Product")

    # Save visualization
    dot.render("hierarchical_classifier_architecture", view=True)
