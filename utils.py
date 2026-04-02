import torch
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertPooler
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform, BertAttention, BertIntermediate, \
    BertOutput
import networkx as nx
import scipy.sparse as sp
import numpy as np
from numpy.linalg import inv
import pandas as pd
import os
import logging
import warnings


warnings.filterwarnings('ignore')

BertLayerNorm = torch.nn.LayerNorm


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEmbeddings(nn.Module):
    """Construct the embeddings from features, wl, position and hop vectors.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.raw_feature_embeddings = nn.Linear(config.x_size, config.hidden_size)
        self.wl_role_embeddings = nn.Embedding(config.max_wl_role_index, config.hidden_size)
        self.inti_pos_embeddings = nn.Embedding(config.max_inti_pos_index, config.hidden_size)
        self.hop_dis_embeddings = nn.Embedding(config.max_hop_dis_index, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, raw_features=None, wl_role_ids=None, init_pos_ids=None, hop_dis_ids=None):
        raw_feature_embeds = self.raw_feature_embeddings(raw_features).unsqueeze(1)
        role_embeddings = self.wl_role_embeddings(wl_role_ids)
        position_embeddings = self.inti_pos_embeddings(init_pos_ids)
        hop_embeddings = self.hop_dis_embeddings(hop_dis_ids)

        # ---- here, we use summation ----
        embeddings = raw_feature_embeds + role_embeddings + position_embeddings + hop_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, residual_h=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                                         encoder_attention_mask)
            hidden_states = layer_outputs[0]

            # ---- add residual ----
            if residual_h is not None:
                for index in range(hidden_states.size()[1]):
                    hidden_states[:, index, :] += residual_h

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


def create_triples(path: str, encoding_str: str = "utf-8"):
    """
    :param path: str
        The path of the file containing the graph in .txt format
    :param encoding_str: str
        The encoding of the graph in .txt format
    :return:
        A list of the triples contained in the graph
    """
    with open(path, encoding=encoding_str) as file:
        lines = []
        for line in file.readlines():
            lines.append(line.strip().split("\t"))

    return lines


# Splits the triple into its elements
def get_triple_elements(triples: list):
    """
    :param triples: list
        A list of the triples contained in the graph
    :return:
        heads: list
        A list of the head elements of the triples
        relations: list
        A list of the relation elements of the triples
        tails: list
        A list of the tail elements of the triples
    """
    heads = [triple[0] for triple in triples]
    relations = [triple[1] for triple in triples]
    tails = [triple[2] for triple in triples]

    return heads, relations, tails


def compute_intimacy_matrix(graph: nx.Graph):
    """"
    Args
        graph (nx.Graph): A structure representing the knowledge graph

    Returns
        Intimacy matrix as a numpy array rounded to the third decimal.
    """
    # Compute the adjacency matrix of the graph
    adj_matrix = nx.to_scipy_sparse_array(graph, format="coo", dtype=np.float32).toarray()
    # Compute the sum of every row in the graph
    rowsum = np.array(adj_matrix.sum(1))
    # Compute the inverse square root of every row's sum
    r_inv = np.power(rowsum, -0.5).flatten()
    # Where the r_inv equals infinity (nodes without neighbors) set it to zero
    r_inv[np.isinf(r_inv)] = 0.
    # A diagonal metrix is created using the r_inv as its diagonal
    r_mat_inv = sp.diags(r_inv)
    # Normalized adjacency matrix (D^-0.5 x adj_matrix x D^-0.5) @ means matrix multiplication
    norm = r_mat_inv @ adj_matrix @ r_mat_inv
    c = 0.15
    # Unit sparse matrix
    I = sp.eye(norm.shape[0], dtype=np.float32)
    # Intimacy score formula
    s = c * inv((I - (1 - c) * norm))
    return np.array(s).round(3)


# Select device
def select_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return torch.device(device)


def _is_hierarchical_relation(graph_name: str):
    """"
    A function to determine whether a relation is hierarchical.
    Args:
        graph_name (str): The name of the graph.

        relation_uri (str): The URI of the relation.
    Returns:
        is_hierarchical (bool): Whether the relation is hierarchical.
    """
    import pandas as pd
    import json
    if graph_name.lower() == "agrovoc":
        relation_definitions = pd.read_excel("relation_definitions.xlsx", sheet_name=f"AGROVOC-all")
    elif graph_name.lower() == "foodon":
        relation_definitions = pd.read_excel("relation_definitions.xlsx", sheet_name=f"FoodOn-all")
    else:
        raise Exception(f"Unknown graph name {graph_name}")

    relations = set(relation_definitions["relation type URI"].tolist())
    relation_cats = {}
    for relation_uri in relations:
        relation_categories = relation_definitions.loc[
            relation_definitions['relation type URI'] == relation_uri, "category"].tolist()
        relation_cats[relation_uri] = relation_categories

    with open("relation_categories.txt", "w", encoding="utf8") as f:
        json.dump(relation_cats, f)


def create_nx_graph(path_to_triples: str) -> nx.Graph:
    """

    Args:
        path_to_triples:
            The path to the triples file to create the graph
    Returns:
        graph:
            A networkX graph representation of the triples file.
    """

    graph = nx.DiGraph()
    path = os.path.dirname(path_to_triples)
    relations_mapping = relations_to_ids(os.path.join(path, "relations.txt"))
    with open(path_to_triples, "r", encoding="utf8") as infile:
        for line in infile:
            head, relation, tail = line.strip().split("\t")
            graph.add_edge(head, tail, relation=relations_mapping[relation])

    return graph


def relations_to_ids(path_to_relations):
    relations_mapping = {}
    with open(path_to_relations, "r", encoding="utf8") as infile:
        for index, line in enumerate(infile):
            relations_mapping[line.strip()] = index
    infile.close()
    return relations_mapping


def get_logger():
    # Create logger and set the lowest loger level
    logger = logging.getLogger("evaluation_logger")
    logger.setLevel(logging.DEBUG)

    # Create a filehandler for the logger and the format of every entry
    handler = logging.FileHandler("eval_logger_agrovoc_augmented_scientific.log")

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%d-%m-%y %H:%M")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def id_to_row(entities: torch.Tensor, mapping_tensor_size: int = None):
    """
    Args:
        entities: Bx1 tensor of type torch.long, where B is the batch size, which
                  contains the entities IDs
        mapping_tensor_size: int, The maximum numerical ID of an entiy.

    Returns:
        id2row: Bx1 tensor of type torch.long, where B is the max_id+1, storing each
        entity's row position into its raw id slot.
    """
    if mapping_tensor_size is None:
        mapping_tensor_size = int(entities.max()) + 1
    id2row = torch.full((mapping_tensor_size,), -99, dtype=torch.long, device=entities.device)
    index = torch.arange(entities.shape[0], device=entities.device)
    id2row = id2row.scatter(0, entities, index)

    return id2row


def compute_ranking(predictions: torch.Tensor, actual_entities: torch.Tensor):
    """
    Args:
        predictions: Tensor containing the predictions of the model based on scoring function
        actual_entities: Tensor containing the actual entities IDs

    Returns:
        average_ranking: The reciprocal rank score of the predictions based on the
        average of optimistic and pessimistic rank
    """

    # Define the position of the true entity in the predictions ranking tensor
    position_of_true_entity = predictions.gather(dim=1, index=actual_entities)
    # Define the number of entities with score higher than the score of the
    # actual entity
    # +1 is used to define the position of the entity. If the true entity scored in the
    # first position then (predictions > position_of_true_entity) would be 0 so +1 marks it
    # is the first one
    optimistic_ranking = (predictions > position_of_true_entity).sum(dim=1, keepdim=True) + 1
    # Define teh number of entities with score greater or equal with the score of the actual entity.
    # If the output is 5 then it means that the true entity ranks fifth in the sorted list
    pessimistic_ranking = (predictions >= position_of_true_entity).sum(dim=1, keepdim=True)
    # Compute the average rank
    average_ranking = (optimistic_ranking + pessimistic_ranking) / 2

    return average_ranking


def head_tail_filters(triples_to_evaluate: torch.Tensor, number_of_entities, graph, raw_id_to_embedding_row):
    """
    Args:
        triples_to_evaluate: Bx3 tensor of type torch.long, where B is the batch size
        number_of_entities: int, number of entities
        graph: nx.MultiDiGraph containing all edges used to filter candidates
        raw_id_to_embedding_row: A tensor mapping every entity id in the correct embedding row

    Returns:
        heads_mask: A tensor containing the mask to find all the true triples where every head appears
        in the graph
        tails_mask: A tensor containing the mask to find all the true triples where every tail appears
        in the graph
    """
    device = triples_to_evaluate.device
    number_of_triples = triples_to_evaluate.shape[0]
    heads_mask = torch.zeros((number_of_triples, number_of_entities), dtype=torch.bool, device=device)
    tails_mask = torch.zeros((number_of_triples, number_of_entities), dtype=torch.bool, device=device)

    triples = triples_to_evaluate.tolist()
    for index, (head, tail, relation) in enumerate(triples):
        head_edges = graph.out_edges(head, data="weight")
        tail_edges = graph.in_edges(tail, data="weight")
        for _, t, rel in head_edges:
            if rel == relation and t != tail:
                tail_id = raw_id_to_embedding_row[t]
                if tail_id != -99:
                    tails_mask[index, tail_id] = 1
        for h, _, rel in tail_edges:
            if rel == relation and h != head:
                head_id = raw_id_to_embedding_row[h]
                if head_id != -99:
                    heads_mask[index, head_id] = 1

    return heads_mask, tails_mask


def compute_number_of_inductive_triples(triples: torch.Tensor, inductive_entities: list):
    count = 0
    for _, (head, tail, _) in enumerate(triples):
        if head in inductive_entities or tail in inductive_entities:
            count += 1
    return count


def split_in_buckets(triples: torch.Tensor,
                     inductive_entities: torch.Tensor,
                     reciprocal_ranks: torch.Tensor,
                     number_of_inductive_triples,
                     check_inductiveness: bool = True):
    device = reciprocal_ranks.device
    # Number of triples equals batch size
    number_of_triples = triples.shape[0]

    # Initialize the bucket tensors
    # index 0: new head
    # index 1: new tail
    # index 2: both new
    batch_direction_reciprocal_ranks = torch.full((number_of_inductive_triples, 4),
                                                  float("nan"), device=device)  # -> index 4: Relation of the triple
    reciprocal_ranks_by_position = torch.zeros((3, 1), device=device)

    items_in_bucket = torch.zeros((3, 1), device=device)
    # batch_ranks_by_position = torch.zeros(number_of_inductive_triples,3)
    inductive_triple_index = 0
    for index, (head, tail, relation) in enumerate(triples):

        head, relation, tail = head.item(), relation.item(), tail.item()

        # Use of the average to examine the inductive nature of the model
        sum_rr = reciprocal_ranks[index] + reciprocal_ranks[index + number_of_triples]

        if head in inductive_entities and tail in inductive_entities:
            batch_direction_reciprocal_ranks[inductive_triple_index][3] = relation
            batch_direction_reciprocal_ranks[inductive_triple_index][2] = sum_rr / 2
            reciprocal_ranks_by_position[2] += sum_rr
            items_in_bucket[2] += 1.0
            inductive_triple_index += 1
        elif head in inductive_entities:
            # Use of the directional reciprocal rank to examine how well the model
            # performs in a retrieval task (Retrieve the correct embedding)
            head_new_rr = reciprocal_ranks[index].item()
            batch_direction_reciprocal_ranks[inductive_triple_index][3] = relation
            batch_direction_reciprocal_ranks[inductive_triple_index][0] = head_new_rr
            reciprocal_ranks_by_position[0] += sum_rr
            items_in_bucket[0] += 1.0
            inductive_triple_index += 1
        elif tail in inductive_entities:
            # Use of the directional reciprocal rank to examine how well the model
            # performs in a retrieval task (Retrieve the correct embedding)
            tail_new_rr = reciprocal_ranks[index + number_of_triples].item()
            batch_direction_reciprocal_ranks[inductive_triple_index][3] = relation
            batch_direction_reciprocal_ranks[inductive_triple_index][1] = tail_new_rr
            reciprocal_ranks_by_position[1] += sum_rr
            items_in_bucket[1] += 1.0
            inductive_triple_index += 1

    if check_inductiveness:
        return reciprocal_ranks_by_position, items_in_bucket
    else:
        # return the necessary tensors for the bullet three
        return batch_direction_reciprocal_ranks


def compute_macro_mrrs(direction_reciprocal_ranks: torch.Tensor, hierarchical_relations: list):
    device = direction_reciprocal_ranks.device
    macro_mrr_per_rel_category = torch.zeros((2, 3), device=device)
    unique_relations = torch.unique(direction_reciprocal_ranks[:, -1])
    mrr_per_relation = torch.full((len(unique_relations), 4), float("nan"), device=device)
    for index, relation in enumerate(unique_relations):
        relation = relation.item()
        rr = direction_reciprocal_ranks[direction_reciprocal_ranks[:, -1] == relation][:, :3]

        # implement the category check
        # Category IDs 0.0 -> non-hierarchical relation, 1.0 -> hierarchical relation
        category = 1.0 if relation in hierarchical_relations else 0.0
        mrr_per_relation[index, :-1] = torch.nanmean(rr, dim=0)
        mrr_per_relation[index, -1] = category


    macro_mrr_per_bucket = torch.nanmean(mrr_per_relation[:, :-1], dim=0)

    non_hierarchical_relations_rr = mrr_per_relation[mrr_per_relation[:, -1] == 0.0][:, :-1]
    macro_mrr_per_rel_category[0] = torch.nanmean(non_hierarchical_relations_rr, dim=0)
    hierarchical_relations_rr = mrr_per_relation[mrr_per_relation[:, -1] == 1.0][:, :-1]
    macro_mrr_per_rel_category[1] = torch.nanmean(hierarchical_relations_rr, dim=0)

    return macro_mrr_per_bucket, macro_mrr_per_rel_category


def split_params_for_lrs(inductive_link_model):
    text_encoder_params = [param for param in inductive_link_model.text_encoder.parameters() if param.requires_grad]
    graph_encoder_params = list(inductive_link_model.graph_encoder.parameters())

    # Convert parameters to ids
    text_encoder_ids = {id(param) for param in text_encoder_params}
    graph_encoder_ids = {id(param) for param in graph_encoder_params}

    # Fusion and relations embeddings parameters
    remaining_model_params = [param for param in inductive_link_model.parameters() if
                              (id(param) not in text_encoder_ids) and (id(param) not in graph_encoder_ids)]
    remaining_ids = {id(param) for param in remaining_model_params}
    # Ensure that there is no parameters overlap
    assert len(text_encoder_ids.intersection(graph_encoder_ids)) == 0
    assert len(graph_encoder_ids.intersection(remaining_ids)) == 0
    assert len(text_encoder_ids.intersection(remaining_ids)) == 0

    return text_encoder_params, graph_encoder_params, remaining_model_params


def create_params_groups(parameters, lr, weight_decay):
    params_with_no_decay = ("bias", "LayerNorm.weight", "layer_norm", "layernorm")
    decay_params, no_decay_params = [], []

    for name, param in parameters:
        if not param.requires_grad:
            continue

        if any(param_with_no_decay in name for param_with_no_decay in params_with_no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})

    return param_groups


def freeze_text_encoder(text_encoder, num_of_unfrozen_layers: int = 6, freeze_embedding_layer: bool = True):
    if not freeze_embedding_layer:
        for param in text_encoder.embeddings.parameters():
            param.requires_grad = True

    bert = text_encoder.encoder
    text_encoder_layers = len(bert.encoder.layer)

    last_frozen_layer = max(0, (text_encoder_layers - num_of_unfrozen_layers))

    for layer in bert.encoder.layer[:last_frozen_layer]:
        for param in layer.parameters():
            param.requires_grad = False


def remove_bns_from_sampling(uris, uris2ids, device):
    entities_pool = []
    for key in uris:
        if "_:node" not in key:
            entities_pool.append(uris2ids[key])

    return torch.tensor(entities_pool, dtype=torch.long, device=device)


def evaluate_wrong_replacements(triples: torch.Tensor,
                                inductive_entities: torch.Tensor,
                                true_entities: torch.Tensor,
                                predictions: torch.Tensor,
                                hierarchical_relations: torch.Tensor,
                                row2id: torch.Tensor,
                                ids2entities,
                                hierarchy_dict,
                                neighborhood_dict=None
                                ):
    # Define the device, entities and inductive entities in the batch
    device = predictions.device
    batch_size = triples.size(0)
    entities = torch.tensor([int(id) for id in ids2entities.keys()], device=device)
    # Get head, tails of the triples and total entities
    head_tails, rels = torch.chunk(triples, 2, dim=-1)
    entities_in_batch = torch.unique(head_tails)

    # Get the inductive entities in the batch
    inductives_mask = torch.isin(entities_in_batch, inductive_entities)
    rows = torch.nonzero(inductives_mask, as_tuple=False)
    inductive_entities_in_batch = entities_in_batch[rows]

    # Get the triples that contain the inductive entities
    inductive_triples_mask = torch.isin(head_tails.reshape(1, -1), inductive_entities_in_batch)
    _, cols = torch.nonzero(inductive_triples_mask, as_tuple=True)
    inductive_triples_ids = torch.unique(cols // 2)
    triples_with_inductive_entities = triples[inductive_triples_ids]

    # Get the predictions of the inductive triples
    head_predictions = predictions[inductive_triples_ids]
    tail_predictions = predictions[inductive_triples_ids + batch_size]
    triples_inductive_entities_predictions = torch.cat([head_predictions, tail_predictions])

    # Get the head, tails and relations of the inductive triples
    # h,t,r = torch.chunk(triples_with_inductive_entities, 3, dim=-1)
    # entities_in_inductive_triples = torch.cat([h,t])
    true_heads = true_entities[inductive_triples_ids]  # shape [n, 1]
    true_tails = true_entities[inductive_triples_ids + batch_size]
    true_entities_in_inductive_triples = torch.cat([true_heads, true_tails], dim=0)
    # Get the scores of entities in the inductive triples
    # true_entities_in_inductive_triples = true_entities[entities_in_inductive_triples].squeeze(-1)

    scores_of_true_entities = torch.gather(triples_inductive_entities_predictions, dim=1,
                                           index=true_entities_in_inductive_triples)

    wrong_place_scores = triples_inductive_entities_predictions > scores_of_true_entities

    graph_entities = len(set(ids2entities.keys()))

    # First column are heads, second column are tails
    items_in_bucket = torch.zeros((2, 1), device=device)
    new_entities = torch.zeros((triples_with_inductive_entities.shape[0], 2), device=device)
    # new_tails = torch.zeros((triples_with_inductive_entities.shape[0],2), device=device)
    found_hierarchy = torch.zeros((graph_entities), device=device, dtype=torch.int32)

    found_head_hierarchy = torch.zeros((graph_entities), device=device, dtype=torch.int32)
    found_tail_hierarchy = torch.zeros((graph_entities), device=device, dtype=torch.int32)
    found_neighborhood_hierarchical = torch.zeros((graph_entities, 2),
                                                  device=device, dtype=torch.int32)
    found_neighborhood_nonhierarchical = torch.zeros(
        (graph_entities, 2),
        device=device, dtype=torch.int32)
    wrong_placements = torch.zeros((graph_entities), device=device, dtype=torch.int32)
    wrong_head_placements = torch.zeros((graph_entities), device=device, dtype=torch.int32)
    wrong_head_placements_nh = torch.zeros((graph_entities), device=device, dtype=torch.int32)
    wrong_tail_placements = torch.zeros((graph_entities), device=device, dtype=torch.int32)
    wrong_tail_placements_nh = torch.zeros((graph_entities), device=device, dtype=torch.int32)

    id2row_list = row2id.tolist()

    # Split the entities of triples containing inductive entities based on whether
    # the inductive entity is at head or tail position
    for index, (head, tail, relation) in enumerate(triples_with_inductive_entities):
        head, tail, relation = head.item(), tail.item(), relation.item()
        if head in inductive_entities:
            items_in_bucket[0] += 1.0
            new_entities[index][0] = head
        if tail in inductive_entities:
            items_in_bucket[1] += 1.0
            new_entities[index][1] = tail

    # for true_index, true_entity in enumerate(true_inductive_entities_position):
    for true_index, true_entity in enumerate(true_entities_in_inductive_triples):

        # if true_entity.item() == 0:
        #     continue
        # Get the raw id and the uri of the inductive entity

        # true_entity_id = id2row_list.index(true_entity.item())
        true_entity_id = entities[true_entity]
        true_entity_uri = ids2entities[str(true_entity_id.item())]
        inductive_entity = inductive_entities[true_index]
        triples_indices = true_index % triples_with_inductive_entities.shape[0]

        # Store the entities of the hierarchy of the inductive entity
        hierarchy = []
        siblings = set()
        cousins = set()
        neighborhood = set()
        # IF foodon and the inductive entity is a blank node
        if "_:node" in true_entity_uri:
            uri = ids2entities[str(triples_with_inductive_entities[triples_indices][0].item())]
            for key, h in hierarchy_dict.items():
                if uri in h:
                    hierarchy.extend(h)
                    siblings.update(neighborhood_dict[uri][key]["siblings"])
                    cousins.update(neighborhood_dict[uri][key]["cousins"])
                    neighborhood.update(neighborhood_dict[uri][key]["neighbors"])
        else:
            for key, h in hierarchy_dict.items():
                if true_entity_uri in h:
                    hierarchy.extend(h)
                    siblings.update(neighborhood_dict[true_entity_uri][key]["siblings"])
                    cousins.update(neighborhood_dict[true_entity_uri][key]["cousins"])
                    neighborhood.update(neighborhood_dict[true_entity_uri][key]["neighbors"])

        # Get the entities that were scored higher than the actual inductive entities
        preds = torch.nonzero(wrong_place_scores[true_index])
        wrong_placements[true_entity_id] += preds.shape[0]
        if true_entity_id in triples_with_inductive_entities[triples_indices, 0]:
            wrong_head_placements[true_entity_id] += preds.shape[0]
        if true_entity_id in triples_with_inductive_entities[triples_indices, 1]:
            wrong_tail_placements[true_entity_id] += preds.shape[0]

        for index, prediction in enumerate(preds):
            # predicted_entity_id = id2row_list.index(prediction.item())
            predicted_entity_id = entities[prediction]
            predicted_entity_uri = ids2entities[str(predicted_entity_id.item())]
            if predicted_entity_uri in hierarchy:
                found_hierarchy[true_entity_id] += 1
                if triples_with_inductive_entities[triples_indices, 2] in hierarchical_relations:
                    if true_entity_id in triples_with_inductive_entities[triples_indices, 0]:
                        found_head_hierarchy[true_entity_id] += 1
                        if predicted_entity_uri in siblings:
                            found_neighborhood_hierarchical[true_entity_id, 0] += 1
                    if true_entity_id in triples_with_inductive_entities[triples_indices, 1]:
                        found_tail_hierarchy[true_entity_id] += 1
                        if predicted_entity_uri in cousins:
                            found_neighborhood_hierarchical[true_entity_id, 1] += 1
                else:
                    if true_entity_id in triples_with_inductive_entities[triples_indices, 0]:
                        found_head_hierarchy[true_entity_id] += 1
                        if predicted_entity_uri in neighborhood:
                            found_neighborhood_nonhierarchical[true_entity_id, 0] += 1
                    if true_entity_id in triples_with_inductive_entities[triples_indices, 1]:
                        found_tail_hierarchy[true_entity_id] += 1
                        if predicted_entity_uri in neighborhood:
                            found_neighborhood_nonhierarchical[true_entity_id, 1] += 1

    return wrong_placements, wrong_head_placements, wrong_tail_placements, found_hierarchy, found_head_hierarchy, found_tail_hierarchy, found_neighborhood_hierarchical, found_neighborhood_nonhierarchical

