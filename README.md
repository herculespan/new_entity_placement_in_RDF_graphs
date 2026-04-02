#Inductive Link Prediction with Text + Graph Topology

A transformers based implementation for **inductive link prediction** in RDF graphs that combines: 
- **Text embeddings** from entity descriptions, using BERT encoder 
- **Graph Embeddings** using a GRAPH-BERT based graph encoder
- **Fusion** between textual and graph representations
- Handling of the **inductive (unseen) entities** via a text-to-graph mapper
