# Fusing textual and graph topology embeddings for new entity placement in agrifood-related RDF graphs

## Introduction
We propose an inductive link prediction model that treats the addition and placement of new entities to an RDF graph as a triple plausibility scoring and ranking task. Οur model **fuses textual descriptions of entities** with **graph topology**-**related information** (i.e., information obtained from the graph structure) to generate entity embeddings for the inductive link prediction task. The graphs we used were: (i) the `AGROVOC` **thesaurus**, and (ii) the `FoodOn` **ontology**. 

Text provides information for entities that is **independent of the graph and its topology** and enables the computation of embeddings for entities that are not in the graph. In the case of graph entities, **structural information** obtained from their position in the graph is also an important source of entity-related information to consider. 
For entities not contained in the graph, there are **no graph topology**-**based embeddings** available. In that case, **we use graph**-**like embeddings** created from the entities' textual descriptions.

Our experiments were based on **setups** using (i) entity descriptions obtained from texts of **different sources** and **lengths** (entity definitions vs abstracts of scientific publications vs non-scientific textual descriptions), and (ii) **two variants** of the `AGROVOC` and `FoodOn` graphs (entity-only vs augmented graph variant) regarding the use of triples conveying entity-related metadata information. 

To optimize entity and relation embeddings, we use a **distance**-**based scoring function** ([TransE](https://dl.acm.org/doi/10.5555/2999792.2999923)) assigning high scores to positive examples and low scores to negative examples. The learning objective is to **minimize the loss** computed with a [margin-based loss function](https://ceur-ws.org/Vol-2377/paper_1.pdf). 

## Model
Our model is composed of (i) a **text encoder**, (ii) a **graph encoder**, (iii) a **fusion component**, and (iv) an **inductive link prediction component**. Each component is built as PyTorch `nn.Module`. 

<img width="7480" height="4068" alt="Figure_4" src="https://github.com/user-attachments/assets/00276c25-4120-4da4-bd7e-5cdba0a8c2f7" />

### Text encoder
We use [BERT](https://arxiv.org/abs/1810.04805) with the **configuration it ships with** in its `BASE` version: number of stacked encoder layers = $12$, hidden layer size = $768$, and number of self-attention heads = $12$. 

The **maximum sequence length** is set to $64$ for token sequences generated from entity definitions and $128$ for scientific and non-scientific textual descriptions. We use the embedding of the `[CLS]` token as the text-based entity embedding.

### Graph encoder
Our graph encoder is built with [Graph-BERT](https://arxiv.org/abs/2001.05140), a Graph Neural Network that generates **embeddings of graph entities** based on the **attention mechanism**. Graph-BERT consists of (i) an **embedding layer**, (ii) the **encoder layers**, and (iii) a **pooling layer**. 

The **embedding layer** computes initial entity embeddings by summing the embeddings from each of the 4 different types of input provided to Graph-BERT (raw features of entities, Weisfeiler-Lehman IDs intimacy-based relative position IDs, and hop distance IDs). 

The initially computed embeddings are the input to the **encoder layer**, which creates sequences of contextualized hidden states for each entity. 

The **pooling layer** takes as input each sequence of contextualized hidden states and returns a single embedding ($dim=64$) for each entity (similar to the embedding of the special `[CLS]` token outputted from the last hidden state of the text encoder).

We use the following **configuration** for Graph-BERT: number of intimate neighbors of a graph entity = $5$, number of stacked encoder layers = $2$, hidden layer size = $64$, number of self-attention heads = $4$.

### Fusion component
It **combines** the text- and graph topology-based embeddings into a **single vector representation** for each entity. Its purpose is to learn the gate $\alpha$ for weighing the importance of text compared to that of graph topology when constructing the final entity representation. 

The fusion component consists of (i) a stack of **fully connected layers** that project the text- and graph-based embeddings into the common vector space where the fused embeddings live ($dim=192$), (ii) a stack of **vector normalization layers**, (iii) a stack of **dropout layers**, and (iv) a **gating layer** that learns $\alpha$ (fully connected layer). 

The **gating layer** takes as input a vector that contains (i) the text-based embedding, (ii) the graph-based embedding, and (iii) a scalar (gate hint) that indicates whether the graph-based embedding is computed from the graph or predicted from text. Its output is passed through the `sigmoid` activation to become a weighting coefficient $\in \left(0,1\right)$. 

By initializing the gating layer with **zero weights** and **bias** = $1$, we nudge text by giving $\alpha$ the starting value $0.73 = sigmoid\left(1\right)$.

### Inductive link prediction component
The inductive link prediction component consists of (i) a **text**-**to**-**graph mapper**, (ii) a **relation embedding layer**, and (iii) a **fully connected layer** projecting relation embeddings into the common vector space. 

It is **responsible for the scoring** of positive and negative triples using TransE and computes the loss based on the margin loss. It also provides a **gate hint value** per entity indicating whether the graph embedding is outputted by the graph encoder ($gate\ hint = 1.0$) or produced by the text-to-graph mapper ($0.2 ≤ gate\ hint ≤ 0.5$). The gate hint value quantifies **how much the graph**-**like embeddings** predicted from text **should be trusted**. 

The **text**-**to**-**graph mapper** is a Multi-Layer Perceptron (`MLP`) that generates graph-like embeddings for new/unseen entities, using their text-based embeddings. It consists of two linear layers and a `GELU` activation function between them. Its output dimension is set equal to the dimensionality of the graph embedding space.

The **relation embedding layer** creates relation embeddings through the encoding of relation definitions with BERT. The embeddings of the `[CLS]` tokens are projected onto the common vector space by the linear projection layer. Relation embeddings are finally available as **ID**-**indexed vectors** after copying the output of the projection layer into a trainable lookup table.

This project is licensed under the MIT License. See the `LICENSE` file for details.
