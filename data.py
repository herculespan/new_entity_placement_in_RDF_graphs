from utils import create_triples, get_triple_elements, compute_intimacy_matrix, remove_bns_from_sampling
import numpy as np
from torch.utils.data import Dataset
import os
import torch
import json
from transformers import BertTokenizer
from typing import List, Dict
import networkx as nx
from collections import Counter
import hashlib
from config import GraphBertConfig


# AGROVOC config
# MAX_ENTITIES_NUMBER = 38090
# MAX_RELATIONS_NUMBER = 49
# FoodOn config
MAX_ENTITIES_NUMBER = 62104
MAX_RELATIONS_NUMBER = 19


def load_dictionary(path:str):
    with open(path, 'r') as infile:
        return json.load(infile)


def load_triples(path:str, entities_mapping:dict, relations_mapping:dict):
    entities = set()
    entities_ids = set()
    relations = []
    link_list = []
    with open(path, "r", encoding="utf-8") as infile:
        triples = []

        for line in infile.readlines():
            triple = line.strip().split("\t")
            head = triple[0]
            relation = triple[1]
            tail = triple[2]
            entities.update([head, tail])
            entities_ids.update([entities_mapping[head], entities_mapping[tail]])
            link_list.append((head, tail))
            relations.append(relation)
            triples.append([entities_mapping[head], entities_mapping[tail], relations_mapping[relation]])
    return entities, relations, triples, link_list, entities_ids


def _get_entity_metadata(graph_name: str, entity: str, labels_dict: dict) -> dict:
    """"
    Args
        graph_name:
            The name of the graph under examination.
        entity:
            The id of the entity under examination.

    Returns

        metadata:
            A dictionary containing the entity label, type and top concepts.
    """

    # Check in which graph the entity appears and get the label of the entity
    if "agrovoc" in graph_name.lower():
        path = "Datasets/agrovoc_2020_concepts/agrovoc_2020_en_clean.txt"
        label = labels_dict[entity]
        with open("Datasets/agrovoc_2020_concepts/hierarchy.txt") as infile:
            hierarchy = json.load(infile)
    elif "foodon" in graph_name.lower():
        path = "Datasets/foodon_classes_0209/foodon_full_0209_clean.txt"
        label = labels_dict[entity]
        with open("Datasets/foodon_classes_0209/hierarchy.txt") as infile:
            hierarchy = json.load(infile)
    else:
        raise ValueError("Unknown graph_name: must contain 'agrovoc' or 'foodon'")

    # Initialize metadata with the first one being its label
    metadata = {entity: [label]}
    # Get the top concept and the type of the entity from its graph
    triples = create_triples(path)
    heads, relations, tails = get_triple_elements(triples)
    # Get the indices where the entity under examination is the head of
    indices = np.where(np.array(heads) == entity)[0]

    # Get the type of the entity based on the rdf:type property
    rdf_iri = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    entity_type = [tails[index] for index in indices if relations[index] == rdf_iri]
    metadata[entity].append(entity_type[0])

    # Get the top concept (root of the hierarchy it belongs to) of the entity under examination
    top_concept = set()
    for concept_uri, hierarchy_list in hierarchy.items():
        if entity in hierarchy_list or entity == concept_uri:
            top_concept.add(concept_uri)

    metadata[entity].extend(top_concept)
    return metadata


def map_entities(filepath: str):
    """
    Args:
        filepath (str): A filepath to the file containing triples.
    Returns:
        dict_ents (dict): A  dictionary with mappings of string descriptions to unique
                    integer IDs.
        id_ents (dict): A dictionary with mapping of ids to string descriptions
    """
    triples = create_triples(filepath)
    heads, _, tails = get_triple_elements(triples)

    entities = set(heads + tails)
    dict_ents = {}
    id_ents = {}
    for index, entity in enumerate(entities):
        dict_ents[entity] = index
        id_ents[int(index)] = entity
    return dict_ents, id_ents




def _in_batch_negative_sampling(positive_pairs, num_negative_samples, device="cpu", entities_to_mask_out=None):
    # Get the batch size length
    batch_size = positive_pairs.size(0)
    # Choose if head or tail will be replaced (0 -> head,1 -> tail)
    head_or_tail = torch.randint(0, 2, [batch_size * num_negative_samples], device=device)
    # Place all the entities in batch size rows where in every row the pattern is head0-tail0-head1-tail1...
    flattened_entities = torch.flatten(positive_pairs).repeat(batch_size, 1).float()
    # Entities in the batch
    entities = flattened_entities[0].clone().long()

    # Number of entities contained in batch
    number_of_entities = batch_size * 2
    # Entities indices in the batch in a shape [batch_size,2]
    idx = torch.arange(number_of_entities).reshape(batch_size, 2)
    zeros = torch.zeros((batch_size, 2), )
    # Replace with zero the positive pair head tail combination in every row of flattened entities tensor
    flattened_entities.scatter_(1, idx,
                                zeros)  # Done to ensure that the negative sample is created from entities of different rows

    if entities_to_mask_out is not None:
        mask = torch.isin(entities, entities_to_mask_out)
        extended_mask = mask.repeat(batch_size, 1)
        flattened_entities[~extended_mask] = 0.0

    # Returns a tensor where each row contains num_negatives * repeats index samples from a multinomial distribution
    random_idx = flattened_entities.multinomial(num_negative_samples, replacement=True)

    # Make the line columns and then the vector 1D
    random_idx = random_idx.t().flatten().long()

    # Make the indices entities ids
    sampled_entities = entities[random_idx]

    # Start with as many copies of the positive head tail pairs as the number of negative samples
    neg_sample = positive_pairs.repeat((num_negative_samples, 1)).to(device)

    # For each of the batch_size * num_of_negative_samples rows, replace either the head or tail
    # with the id of the entity selected as negative sample
    row = torch.arange(batch_size * num_negative_samples, device=device)
    neg_sample[row, head_or_tail] = sampled_entities

    # Reshape the negative sample back to [num_of_negative_samples,batch_size,2]
    neg_sample = neg_sample.reshape(-1, batch_size, 2)
    # Transpose to [batch_size,num_of_negative_samples,2] which is the shape of the returned
    # tensor containing the negative samples.
    neg_sample.transpose_(0, 1)

    return neg_sample, head_or_tail


def sanitize_negative_samples_mixed(
        negative_samples: torch.Tensor,  # [batch_size,number_of_negative_samples,2]
        pos_pairs: torch.Tensor,  # [B,2] (positive heads/tails)
        relations: torch.Tensor,  # [B,1] or [batch_size]
        graph_ids: torch.Tensor,  # [num_triples] encoded IDs
        global_entity_pool: torch.Tensor,  # [E]
        # is_neighborhood: torch.Tensor,  # [batch_size,number_of_negative_samples] bool
        corruption_side: torch.Tensor,  # [batch_size,number_of_negative_samples] 0=head,1=tail  (recommended)
        head_offsets: torch.Tensor = None,  # [num_entities, num_rel, 2]
        tail_offsets: torch.Tensor = None,  # [num_entities, num_rel, 2]
        head_candidates: torch.Tensor = None,  # [flat]
        tail_candidates: torch.Tensor = None,  # [flat]
        max_iters: int = 10
) -> torch.Tensor:
    """
    Enforces simultaneously:
      (a) no cycles (h != t)
      (b) not present in original graph (not a false negative)
      (c) unique negative triples per positive row (no duplicates within row)

    Resampling policy:
      - If sample is neighborhood-based:
          resample head from head_candidates[ pos_head, rel ]
          resample tail from tail_candidates[ pos_tail, rel ]
        with fallback to global pool if neighborhood pool length is 0.
      - If sample is global-based: resample from global_entity_pool.
    """

    device = negative_samples.device
    batch_size, number_of_negative_samples, _ = negative_samples.shape

    rel = relations.squeeze(-1) if relations.dim() == 2 else relations  # [batch_size]
    pos_head = pos_pairs[:, 0]  # [batch_size]
    pos_tail = pos_pairs[:, 1]  # [batch_size]

    total_entities = global_entity_pool.shape[0]

    def triple_ids(triples: torch.Tensor) -> torch.Tensor:
        # triples: [batch_size,number_of_negative_samples,2] -> ids: [batch_size,number_of_negative_samples]
        relation = rel.unsqueeze(1).expand(batch_size,
                                           number_of_negative_samples)  # [batch_size,number_of_negative_samples]
        return (triples[..., 0] * MAX_RELATIONS_NUMBER + relation) * MAX_ENTITIES_NUMBER + triples[..., 1]

    def _sample_global(num: int) -> torch.Tensor:
        idx = torch.randint(0, total_entities, (num,), device=device)
        return global_entity_pool[idx]

    def _resample_entities(batch_index: torch.Tensor, negative_sample_index: torch.Tensor, side: torch.Tensor) -> None:
        """
        Resample negative_samples[batch_index, negative_sample_index, side] in-place,
        choosing pool based on is_neighborhood[batch_index,negative_sample_index] and side.
        """
        if batch_index.numel() == 0:
            return

        negative_samples[batch_index, negative_sample_index, side] = _sample_global(batch_index.numel())

    for _ in range(max_iters):
        changed = False

        # 1) cycles: h == t
        cyc = (negative_samples[..., 0] == negative_samples[..., 1])  # [batch_size,number_of_negative_samples]
        if cyc.any():
            cycles = cyc.nonzero(as_tuple=False)  # [K,2]
            batch_index, negative_sample_index = cycles[:, 0], cycles[:, 1]
            side = corruption_side[batch_index, negative_sample_index]
            _resample_entities(batch_index, negative_sample_index, side)
            changed = True

        # 2) false negatives: triple exists in original graph
        ids = triple_ids(negative_samples).reshape(-1)  # [B*N]
        bad = torch.isin(ids, graph_ids)
        if bad.any():
            bad_idx = bad.nonzero(as_tuple=False).squeeze(1)  # [K]
            batch_index = bad_idx // number_of_negative_samples
            negative_sample_index = bad_idx % number_of_negative_samples
            side = corruption_side[batch_index, negative_sample_index]
            _resample_entities(batch_index, negative_sample_index, side)
            changed = True

        # 3) uniqueness within each row (avoid duplicate triple IDs)
        batch_triples_ids = triple_ids(negative_samples)  # [batch_size,number_of_negative_samples]
        for index in range(batch_size):
            row = batch_triples_ids[index]
            uniques, counts = torch.unique(row, return_counts=True)
            dup_vals = uniques[counts > 1]
            if dup_vals.numel() == 0:
                continue

            for row_value in dup_vals:
                indices = (row == row_value).nonzero(as_tuple=False).squeeze(1)
                idx = indices[1:]  # keep first occurrence
                if idx.numel() == 0:
                    continue
                batch_index = torch.full((idx.numel(),), index, device=device, dtype=torch.long)
                negative_sample_index = idx
                side = corruption_side[batch_index, negative_sample_index]
                _resample_entities(batch_index, negative_sample_index, side)
                changed = True

        if not changed:
            break



    return negative_samples




class TextWithGraphTopologyDataset(Dataset):

    def __init__(self, path_to_triples: str, graph_name: str, graph, tokenizer, descriptions_filename: str,
                 create_mapping: bool, max_length = 64,num_of_negative_samples: int = None,
                 max_neighbors: int = 50, centrality_cache: bool = True):


        # Extract the graph path and map its entities to ids and vice versa
        self.graph_directory_path = os.path.dirname(path_to_triples)
        computed_features = [True for item in os.listdir(self.graph_directory_path) if item.endswith(".pt")]

        if create_mapping:
            if "agrovoc" in graph_name.lower():
                self.entities2ids, self.ids2entities = map_entities(
                    os.path.join(self.graph_directory_path, "agrovoc_graph.txt"))
            elif "foodon" in graph_name.lower():
                self.entities2ids, self.ids2entities = map_entities(os.path.join(self.graph_directory_path, "foodon_graph.txt"))


        else:
            self.entities2ids = load_dictionary(os.path.join(self.graph_directory_path, "entities_mapping.txt"))
            self.ids2entities = load_dictionary(os.path.join(self.graph_directory_path, "ids_mapping.txt"))

        self.relations2ids = load_dictionary(os.path.join(self.graph_directory_path, "relations_mapping.txt"))
        self.ids2relations = load_dictionary(os.path.join(self.graph_directory_path, "ids_relations_mapping.txt"))
        self.labels_dict = load_dictionary(os.path.join(self.graph_directory_path, "entities2labels.txt"))
        # Load the hierarchy and neighborhood dictionary of the graph.
        self.hierarchy = load_dictionary(os.path.join(self.graph_directory_path, "hierarchy.txt"))
        self.relations_properties = load_dictionary(os.path.join(self.graph_directory_path, "relations_properties.txt"))
        # Load the file that contains the triples and extract its entities,relations,and triples
        entities, relations, triples, link_list, entities_ids = load_triples(path_to_triples, self.entities2ids,
                                                                             self.relations2ids)

        self.graph_name = graph_name
        self.device = "cpu"
        self.graph = graph
        self.triples = torch.tensor(triples, dtype=torch.long, device=self.device)
        self.graph_ids = (self.triples[:, 0] * MAX_RELATIONS_NUMBER + self.triples[:, 2]) * MAX_ENTITIES_NUMBER + \
                         self.triples[:, 1]
        self.entities_list = list(self.entities2ids.keys())
        self.entities = torch.tensor([self.entities2ids[entity] for entity in entities], dtype=torch.long,
                                     device=self.device)

        if "foodon" in self.graph_name.lower():
            self.entities_withoutbns = remove_bns_from_sampling(entities, self.entities2ids, self.device)

        self.num_entities = len(entities)
        self.num_relations = len(set(relations))
        self.relation_types = set(relations)
        self.semantic_groups, self.hierarchical_relations = self.create_semantic_groups()
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.relation_description = self.get_relation_description_tokens()
        self.descriptions_filename = descriptions_filename
        self.text_tensor, self.text_mask = self.get_entities_tokens()
        self.num_negative_samples = num_of_negative_samples
        self.is_hierarchical = torch.tensor(self.hierarchical_relations, dtype=torch.long, device=self.device)
        graph_mask_path = os.path.join(self.graph_directory_path, "graph_mask.pt")
        if "train" in path_to_triples:
            self.neighborhood = load_dictionary(os.path.join(self.graph_directory_path, "neighborhoods.txt"))

        if not computed_features:
            # The three following lines ensure that by using the mapping I am moving to the right element in the intimacy matrix
            self.int_graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
            self.graph_mapping = {uri: int_id for uri, int_id in zip(self.graph.nodes(), self.int_graph.nodes())}
            self.int_graph_mapping = {int_id: uri for uri, int_id in zip(self.graph.nodes(), self.int_graph.nodes())}
            self.intimacy_matrix = compute_intimacy_matrix(self.graph)
            self.max_neighbors = max_neighbors
            self.centrality_cache = centrality_cache
            self._centrality_cache = {}
            self.percentile = np.percentile([self.graph.degree(n) for n in self.graph.nodes()], 90)
            self.intimacy_dictionary = self.get_relative_intimacy_scores()
            self.hop_distance_dict = self.get_hop_distance()
            self.wl_role_ids = self.wl_positional_ids()
            self.raw_features, self.wl_role_indices, self.hop_distance_ids, self.intimacy_ids = self.extract_batch_features(
                self.num_entities)
        else:
            self.raw_features = torch.load(os.path.join(self.graph_directory_path, "raw_features_tensor.pt"),
                                           map_location=self.device)
            self.wl_role_indices = torch.load(os.path.join(self.graph_directory_path, "wl_tensor.pt"),
                                              map_location=self.device)
            self.hop_distance_ids = torch.load(os.path.join(self.graph_directory_path, "hop_tensor.pt"),
                                               map_location=self.device)
            self.intimacy_ids = torch.load(os.path.join(self.graph_directory_path, "int_tensor.pt"),
                                       map_location=self.device)
        if os.path.exists(graph_mask_path):
            self.graph_available = torch.load(graph_mask_path, map_location=self.device)
        else:
            n = len(self.ids2entities.keys())
            graph_available = torch.zeros(n, dtype=torch.float32)

            for id in range(n):
                uri = self.ids2entities[str(id)]
                graph_available[id] = 1.0 if uri in self.graph.nodes() else 0.0

            self.graph_available = graph_available
            torch.save(self.graph_available, graph_mask_path)


    def get_relation_description_tokens(self):
        """
        A function to get the descriptions' tokens of relations definitions

        Returns:
            relation_description (torch.Tensor): A tensor of text's descriptions tokens.



        """

        relation_types = self.relations2ids.keys()
        relation_description = torch.zeros(len(relation_types), self.max_length, dtype=torch.long)
        for relation in relation_types:
            definition = self.relations_properties[relation].get("definition", "")
            relation_id = self.relations2ids[relation]
            definition_tokens = self.tokenizer.encode(definition, max_length=self.max_length, return_tensors="pt",
                                                      truncation=True, padding="max_length")
            relation_description[relation_id] = definition_tokens.squeeze(0)

        return relation_description

    def get_entities_tokens(self):
        """
        A function to get the descriptions' tokens and their masks based on a specified file containing dictionaries with
        entities description types (definitions, longer scientific descriptions, non-scientific descriptions).

        Returns:
            text_tensor (torch.Tensor): A tensor of text's descriptions tokens.
            text_masks (torch.Tensor): A tensor of text's descriptions masks.
        """
        path_to_open = os.path.join(self.graph_directory_path, self.descriptions_filename)

        with open(path_to_open, "r", encoding="utf-8") as infile:
            description = json.load(infile)
        infile.close()
        entities = description.keys()
        text_tensor = torch.zeros((len(entities), self.max_length), dtype=torch.long, device=self.device)
        text_masks = torch.zeros((len(entities), self.max_length), dtype=torch.long, device=self.device)
        for id in range(len(entities)):
            try:
                key = self.ids2entities[str(id)]
                text = description[key]
                text_tokens = self.tokenizer.encode(text, max_length=self.max_length, return_tensors="pt",
                                                    truncation=True, padding="max_length").to(self.device)
                text_length = text_tokens.shape[1]
                text_tensor[id, :] = text_tokens
                # text_tensor[id, -1] = text_length
                text_masks[id, :] = (text_tensor[id, :text_length] > 0).float()
            except:
                continue

        return text_tensor, text_masks

    def get_entity_description_tokens_masks(self, batch_ids: torch.Tensor) -> tuple:
        """"
        Get the tokens and masks tensors for every batch id
        """



        batch_tokens = self.text_tensor[batch_ids].to(self.device)
        batch_masks = self.text_mask[batch_ids].to(self.device)

        return batch_tokens, batch_masks

    def get_batch_features(self, batch_ids: torch.Tensor) -> tuple:

        """"
              Get the graph features tensors for every batch id
        """

        raw_features = self.raw_features[batch_ids].to(self.device)
        wl_role_ids = self.wl_role_indices[batch_ids].to(self.device)
        hop_id = self.hop_distance_ids[batch_ids].to(self.device)
        intimacy_ids = self.intimacy_ids[batch_ids].to(self.device)
        graph_mask = self.graph_available[batch_ids].unsqueeze(1).to(self.device)

        return raw_features, wl_role_ids, hop_id, intimacy_ids, graph_mask

    def create_semantic_groups(self):
        """"
        Split the relations types into semantic groups for the computation of graph features

        """

        keys = self.relations_properties.keys()
        semantic_groups = {}
        hierarchical_relations = set()
        for key in keys:
            categories = self.relations_properties[key].get("categories")[0].split(",")
            for category in categories:
                category = category.strip()
                if category not in semantic_groups.keys():
                    semantic_groups[category] = []
                semantic_groups[category].append(key)
        hierarchical_relations.update(
            [self.relations2ids[relation.strip()] for relation in semantic_groups['hierarchical']])

        return semantic_groups, list(hierarchical_relations)

    def get_relative_intimacy_scores(self, number_of_top_k_neighbors: int = 5):
        """"
        Get the top-k intimate neighbors of a node
        """

        # Get the intimacy matrix of the graph
        top_k_neighbors_intimacy_dict = {}
        for node_uri in self.entities2ids:
            if node_uri in self.graph.nodes():
                # Set the intimacy score of the node under examination to -1000
                # node_id = self.entities2ids[node_uri]
                node_id = self.graph_mapping[node_uri]
                s = self.intimacy_matrix[node_id]
                s[node_id] = -1000.0
                # Find the top K neighbors based on intimacy of the node
                top_k_neighbor_index = s.argsort()[-number_of_top_k_neighbors:][::-1]
                top_k_neighbors_intimacy_dict[node_id] = []
                for neighbor_index in top_k_neighbor_index:
                    # neighbor_id = id_to_node_mapping[neighbor_index]
                    top_k_neighbors_intimacy_dict[node_id].append((int(neighbor_index), float(s[neighbor_index])))
        return top_k_neighbors_intimacy_dict

    def get_hop_distance(self):
        """
        For every intimate neighbor, get the shortest distance from the node of interest
        """
        hop_dict = {}
        for node in self.intimacy_dictionary:
            if node not in hop_dict: hop_dict[node] = {}
            for neighbor, score in self.intimacy_dictionary[node]:
                try:
                    hop = int(nx.shortest_path_length(self.graph, source=self.int_graph_mapping[node],
                                                      target=self.int_graph_mapping[neighbor]))
                except:
                    hop = 100
                hop_dict[node][neighbor] = hop
        return hop_dict

    def wl_positional_ids(self):
        """
        Get the Weistfeiler-lehmann roles for evey node
        """
        max_iter = 2
        node_color_dict = {}
        node_neighbor_dict = {}
        for node in self.graph.nodes():
            # node = self.entities2ids[node]
            node = self.graph_mapping[node]
            node_color_dict[node] = 1
            node_neighbor_dict[node] = {}

        for pair in self.graph.edges():
            u1, u2 = pair
            u1, u2 = self.graph_mapping[u1], self.graph_mapping[u2]
            # u1, u2 = self.entities2ids[u1], self.entities2ids[u2]
            if u1 not in node_neighbor_dict:
                node_neighbor_dict[u1] = {}
            if u2 not in node_neighbor_dict:
                node_neighbor_dict[u2] = {}
            node_neighbor_dict[u1][u2] = 1
            node_neighbor_dict[u2][u1] = 1

        iteration_count = 1
        while True:
            new_color_dict = {}
            for node in self.graph.nodes():
                # node = self.entities2ids[node]
                node = self.graph_mapping[node]
                neighbors = node_neighbor_dict[node]
                neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
                color_string_list = [str(node_color_dict[node])] + sorted(
                    [str(color) for color in neighbor_color_list])
                color_string = "_".join(color_string_list)
                hash_object = hashlib.md5(color_string.encode())
                hashing = hash_object.hexdigest()
                new_color_dict[node] = hashing
            color_index_dict = {k: v + 1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
            for node in new_color_dict:
                new_color_dict[node] = color_index_dict[new_color_dict[node]]
            if node_color_dict == new_color_dict or iteration_count == max_iter:
                return new_color_dict
            else:
                node_color_dict = new_color_dict
            iteration_count += 1

    def extract_topology_features(self, entity_id, graph: nx.Graph) -> np.ndarray:
        """
        Extract comprehensive topology features for an entity.

        Args:
            entity_id: Target entity identifier
            graph: Knowledge graph as NetworkX graph
            # entity_metadata: Metadata for entities {entity_id: {type, attributes, ...}}

        Returns:
            Numpy array of topology features
        """

        # entity_id  = self.ids2entities[entity_id]
        if entity_id not in graph:
            return np.zeros(80)  # Return zero vector for missing entities

        entity_metadata = _get_entity_metadata(self.graph_name, entity_id, self.labels_dict)
        features = []

        # 1. Basic connectivity features
        basic_features = self._extract_basic_connectivity(entity_id, graph)
        features.extend(basic_features)  # 10 features

        # 2. Centrality measures
        centrality_features = self._extract_centrality_measures(entity_id, graph)
        features.extend(centrality_features)  # 8 features

        # 3. Neighborhood analysis
        neighborhood_features = self._extract_neighborhood_features(entity_id, graph, entity_metadata)
        features.extend(neighborhood_features)  # 25 features

        # 4. Relation pattern analysis
        relation_features = self._extract_relation_patterns(entity_id, graph)
        features.extend(relation_features)  # 20 features

        # 5. Structural role features
        structural_features = self._extract_structural_roles(entity_id, graph.to_undirected())
        features.extend(structural_features)  # 12 features

        # 6. Path-based features
        path_features = self._extract_path_features(entity_id, graph)
        features.extend(path_features)  # 5 features

        # Ensure exactly 80 features
        features = features[:80]  # Truncate if too many
        features.extend([0.0] * (80 - len(features)))  # Pad if too few

        return np.array(features, dtype=np.float32)

    def _extract_basic_connectivity(self, entity_id: int, graph: nx.Graph) -> List[float]:
        """Extract basic connectivity statistics."""

        # Degree statistics
        degree = graph.degree(entity_id)
        in_degree = graph.in_degree(entity_id) if graph.is_directed() else degree
        out_degree = graph.out_degree(entity_id) if graph.is_directed() else degree

        # Normalized degree (by max possible)
        max_degree = len(graph) - 1
        norm_degree = degree / max_degree if max_degree > 0 else 0

        # Degree ratios
        degree_ratio = in_degree / (out_degree + 1)  # Avoid division by zero

        # Local clustering
        clustering = nx.clustering(graph, entity_id)

        # Triangles
        triangles = sum(1 for _ in range(nx.triangles(graph, entity_id))) if not graph.is_directed() else 0

        # Ego graph size (1-hop and 2-hop neighborhoods)
        ego_1hop = len(set(graph.to_undirected().neighbors(entity_id)))
        ego_2hop_nodes = set()
        for neighbor in graph.neighbors(entity_id):
            ego_2hop_nodes.update(graph.neighbors(neighbor))
        ego_2hop = len(ego_2hop_nodes)

        return [
            np.log(degree + 1),  # Log-normalized degree
            np.log(in_degree + 1),  # Log-normalized in-degree
            np.log(out_degree + 1),  # Log-normalized out-degree
            norm_degree,  # Normalized degree
            degree_ratio,  # In/out degree ratio
            clustering,  # Clustering coefficient
            triangles / (degree + 1),  # Triangle density
            ego_1hop / max_degree,  # 1-hop neighborhood size
            ego_2hop / max_degree,  # 2-hop neighborhood size
            (ego_2hop - ego_1hop) / max_degree  # 2-hop expansion ratio
        ]

    def _extract_centrality_measures(self, entity_id: int, graph: nx.Graph) -> List[float]:
        """Extract centrality measures (with caching for efficiency)."""
        cache_key = f"centrality_{hash(frozenset(graph.edges))}"

        if self.centrality_cache and cache_key in self._centrality_cache:
            centralities = self._centrality_cache[cache_key]
        else:
            # Compute centralities for all nodes
            try:
                betweenness = nx.betweenness_centrality(graph, k=min(100, len(graph)))
                closeness = nx.closeness_centrality(graph)

                if not graph.is_directed():
                    eigenvector = nx.eigenvector_centrality(graph, max_iter=100)
                    pagerank = nx.pagerank(graph, max_iter=100)
                else:
                    eigenvector = nx.eigenvector_centrality(graph, max_iter=100)
                    pagerank = nx.pagerank(graph, max_iter=100)

                centralities = {
                    'betweenness': betweenness,
                    'closeness': closeness,
                    'eigenvector': eigenvector,
                    'pagerank': pagerank
                }

                if self.centrality_cache:
                    self._centrality_cache[cache_key] = centralities

            except Exception as e:
                # logger.warning(f"Centrality computation failed: {e}")
                # Return default values
                return [0.0] * 8

        # Extract centralities for target entity
        betweenness = centralities['betweenness'].get(entity_id, 0)
        closeness = centralities['closeness'].get(entity_id, 0)
        eigenvector = centralities['eigenvector'].get(entity_id, 0)
        pagerank = centralities['pagerank'].get(entity_id, 0)

        # Compute relative rankings
        all_betweenness = list(centralities['betweenness'].values())
        all_closeness = list(centralities['closeness'].values())

        betweenness_rank = (sorted(all_betweenness, reverse=True).index(betweenness) + 1) / len(all_betweenness)
        closeness_rank = (sorted(all_closeness, reverse=True).index(closeness) + 1) / len(all_closeness)

        return [
            betweenness,  # Betweenness centrality
            closeness,  # Closeness centrality
            eigenvector,  # Eigenvector centrality
            pagerank,  # PageRank
            betweenness_rank,  # Betweenness rank (percentile)
            closeness_rank,  # Closeness rank (percentile)
            np.log(pagerank + 1e-10),  # Log PageRank
            (betweenness + closeness + eigenvector) / 3  # Average centrality
        ]

    def _extract_neighborhood_features(self, entity_id: int, graph: nx.Graph,
                                       entity_metadata: Dict[str, List]) -> List[float]:
        """Extract features based on neighborhood analysis."""
        neighbors = list(graph.neighbors(entity_id))[:self.max_neighbors]

        if not neighbors:
            return [0.0] * 25

        # Neighbor type distribution
        neighbor_types = []
        for neighbor in neighbors:
            if neighbor in entity_metadata:
                types = entity_metadata.get(neighbor, ['Unknown'])
                # types = entity_metadata[neighbor].get('types', ['Unknown'])
                neighbor_types.extend(types)

        # Type diversity metrics
        type_counts = Counter(neighbor_types)
        type_entropy = self._compute_entropy(list(type_counts.values()))

        # Most common neighbor types (top 5)
        top_types = [count for _, count in type_counts.most_common(5)]
        top_types.extend([0] * (5 - len(top_types)))  # Pad to 5
        top_types = [count / len(neighbors) for count in top_types]  # Normalize

        # Neighbor degree statistics
        neighbor_degrees = [graph.degree(neighbor) for neighbor in neighbors]
        avg_neighbor_degree = np.mean(neighbor_degrees) if neighbor_degrees else 0
        max_neighbor_degree = max(neighbor_degrees) if neighbor_degrees else 0
        min_neighbor_degree = min(neighbor_degrees) if neighbor_degrees else 0
        degree_variance = np.var(neighbor_degrees) if neighbor_degrees else 0

        # Connectivity patterns within neighborhood
        internal_edges = 0
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1:]:
                if graph.has_edge(n1, n2):
                    internal_edges += 1

        max_internal_edges = len(neighbors) * (len(neighbors) - 1) / 2
        internal_connectivity = internal_edges / max_internal_edges if max_internal_edges > 0 else 0

        # Distance distribution to neighbors
        distances = []
        for neighbor in neighbors[:10]:  # Limit for efficiency
            try:
                dist = nx.shortest_path_length(graph, entity_id, neighbor)
                distances.append(dist)
            except nx.NetworkXNoPath:
                distances.append(float('inf'))

        avg_distance = np.mean([d for d in distances if d != float('inf')]) if distances else 0

        return [
            len(neighbors) / self.max_neighbors,  # Normalized neighbor count
            type_entropy,  # Type diversity entropy
            len(set(neighbor_types)) / len(neighbor_types) if neighbor_types else 0,  # Type uniqueness ratio
            *top_types,  # Top 5 neighbor types (5 features)
            np.log(avg_neighbor_degree + 1),  # Average neighbor degree
            np.log(max_neighbor_degree + 1),  # Max neighbor degree
            np.log(min_neighbor_degree + 1),  # Min neighbor degree
            np.log(degree_variance + 1),  # Neighbor degree variance
            internal_connectivity,  # Internal connectivity ratio
            avg_distance,  # Average distance to neighbors
            len([d for d in distances if d == 1]) / len(distances) if distances else 0,  # Direct connection ratio
            len([d for d in distances if d == 2]) / len(distances) if distances else 0,  # 2-hop connection ratio
            max(neighbor_degrees) / (avg_neighbor_degree + 1) if neighbor_degrees else 0,  # Degree concentration
            len(set(neighbors)) / len(neighbors) if neighbors else 0,  # Neighbor uniqueness
            # Placeholder for remaining features
            0, 0, 0, 0, 0, 0, 0
        ]

    def _extract_relation_patterns(self, entity_id: int, graph: nx.Graph) -> List[float]:
        """Extract patterns from relation types in the neighborhood."""
        relation_types = []
        # Collect all relations
        for neighbor in graph.to_undirected().neighbors(entity_id):
            edge_data = graph.get_edge_data(entity_id, neighbor, {})
            relation = edge_data.get('relation', 'unknown')
            relation_types.append(relation)

        if not relation_types:
            return [0.0] * 20
        # Relation type statistics
        relation_counts = Counter(relation_types)

        relation_entropy = self._compute_entropy(list(relation_counts.values()))

        # Most frequent relations (top 4)
        top_relations = [count for _, count in relation_counts.most_common(4)]
        top_relations.extend([0] * (4 - len(top_relations)))  # Pad to 4
        top_relations = [count / len(relation_types) for count in top_relations]  # Normalize

        # Relation diversity metrics
        unique_relations = len(set(relation_types))
        relation_diversity = unique_relations / len(relation_types)

        # Semantic relation grouping (simplified)
        # semantic_groups = create_semantic_groups(self.graph_name)

        group_counts = {group: 0 for group in self.semantic_groups}

        for relation in relation_types:
            if relation == "unknown":
                continue
            relation = self.ids2relations[str(relation)]
            for group, keywords in self.semantic_groups.items():
                if any(keyword in relation.lower() for keyword in keywords):
                    group_counts[group] += 1
                    break

        group_ratios = [count / len(relation_types) for count in group_counts.values()]
        return [
            unique_relations,  # Number of unique relations
            relation_diversity,  # Relation diversity ratio
            relation_entropy,  # Relation entropy
            *top_relations,  # Top 4 relation frequencies (4 features)
            *group_ratios  # Semantic group ratios (11 features)
        ]

    def _extract_structural_roles(self, entity_id: int, graph: nx.Graph) -> List[float]:
        """Extract structural role features."""
        # Node structural properties

        degree = graph.degree(entity_id)
        neighbors = list(graph.neighbors(entity_id))

        # Role indicators
        is_hub = degree > self.percentile
        is_bridge = self._is_bridge_node(entity_id, graph)
        is_leaf = degree == 1
        is_isolated = degree == 0

        # Structural equivalence (simplified)
        structural_similarity = self._compute_structural_similarity(entity_id, graph)

        # K-core decomposition
        try:
            k_core = nx.core_number(graph).get(entity_id, 0)
            max_k_core = max(nx.core_number(graph).values()) if graph.nodes() else 1
            normalized_k_core = k_core / max_k_core
        except:
            k_core = normalized_k_core = 0

        # Motif participation (simplified - triangles and squares)
        triangle_count = nx.triangles(graph, entity_id) if not graph.is_directed() else 0
        square_count = self._count_squares(entity_id, graph)

        # Ego graph properties
        ego_graph = nx.ego_graph(graph, entity_id, radius=1)
        ego_density = nx.density(ego_graph) if ego_graph.nodes() else 0
        ego_diameter = nx.diameter(ego_graph) if nx.is_connected(ego_graph) and len(ego_graph) > 1 else 0

        return [
            float(is_hub),  # Hub indicator
            float(is_bridge),  # Bridge indicator
            float(is_leaf),  # Leaf indicator
            float(is_isolated),  # Isolated indicator
            structural_similarity,  # Structural similarity score
            normalized_k_core,  # Normalized k-core number
            triangle_count / (degree + 1),  # Triangle participation rate
            square_count / (degree + 1),  # Square participation rate
            ego_density,  # Ego graph density
            ego_diameter,  # Ego graph diameter
            len(ego_graph) / len(graph),  # Ego graph size ratio
            nx.average_clustering(ego_graph) if ego_graph.nodes() else 0  # Ego clustering
        ]

    def _extract_path_features(self, entity_id: int, graph: nx.Graph) -> List[float]:
        """Extract path-based features."""
        # Sample a subset of nodes for efficiency
        sample_nodes = list(graph.nodes())[:100] if len(graph) > 100 else list(graph.nodes())

        distances = []
        for target in sample_nodes:
            if target != entity_id:
                try:
                    dist = nx.shortest_path_length(graph, entity_id, target)
                    distances.append(dist)
                except nx.NetworkXNoPath:
                    distances.append(float('inf'))

        # Filter out infinite distances
        finite_distances = [d for d in distances if d != float('inf')]

        if not finite_distances:
            return [0.0] * 5

        avg_distance = np.mean(finite_distances)
        max_distance = max(finite_distances)
        reachability = len(finite_distances) / len(distances)

        # Distance distribution
        distance_counts = Counter(finite_distances)
        distance_entropy = self._compute_entropy(list(distance_counts.values()))

        # Eccentricity (max distance from node)
        eccentricity = max_distance if finite_distances else 0

        return [
            avg_distance,  # Average shortest path length
            max_distance,  # Maximum shortest path length
            reachability,  # Reachability ratio
            distance_entropy,  # Distance distribution entropy
            eccentricity  # Node eccentricity
        ]

    def _compute_entropy(self, values: List[float]) -> float:
        """Compute Shannon entropy of a list of values."""
        if not values or sum(values) == 0:
            return 0.0

        total = sum(values)
        probabilities = [v / total for v in values if v > 0]
        return -sum(p * np.log2(p) for p in probabilities if p > 0)

    def _is_bridge_node(self, entity_id: int, graph: nx.Graph) -> bool:
        """Check if removing the node would disconnect the graph."""
        if graph.degree(entity_id) < 2:
            return False

        # Simple heuristic: check if removing node increases connected components
        original_components = nx.number_connected_components(graph.to_undirected())
        temp_graph = graph.copy()
        temp_graph.remove_node(entity_id)
        new_components = nx.number_connected_components(temp_graph.to_undirected())

        return new_components > original_components

    def _compute_structural_similarity(self, entity_id: int, graph: nx.Graph) -> float:
        """Compute structural similarity with other nodes."""
        entity_neighbors = set(graph.neighbors(entity_id))

        similarities = []
        for other_node in list(graph.nodes())[:50]:  # Sample for efficiency
            if other_node != entity_id:
                other_neighbors = set(graph.neighbors(other_node))

                # Jaccard similarity of neighborhoods
                intersection = len(entity_neighbors & other_neighbors)
                union = len(entity_neighbors | other_neighbors)

                if union > 0:
                    similarity = intersection / union
                    similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def _count_squares(self, entity_id: int, graph: nx.Graph) -> int:
        """Count 4-cycles (squares) involving the entity."""

        neighbors = list(graph.neighbors(entity_id))
        square_count = 0

        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1:]:
                # Find common neighbors of n1 and n2 (excluding entity_id)
                n1_neighbors = set(graph.neighbors(n1)) - {entity_id}
                n2_neighbors = set(graph.neighbors(n2)) - {entity_id}
                common_neighbors = n1_neighbors & n2_neighbors
                square_count += len(common_neighbors)

        return square_count

    def extract_batch_features(self, batch_ids: int):
        # ids = batch_ids.flatten().tolist()

        raw_features_list = []
        role_ids_list = []
        position_ids_list = []
        hop_ids_list = []
        num_of_neighbors = GraphBertConfig(output_hidden_states=True, output_attentions=True).k
        number_of_raw_features = GraphBertConfig(output_hidden_states=True, output_attentions=True).x_size
        for node_id in range(batch_ids):
            node_uri = self.ids2entities[str(node_id)]
            if node_uri in self.graph.nodes:
                # node_id = self.entities2ids[node_uri]
                raw_feature = self.extract_topology_features(node_uri, self.graph)
                node_id = self.graph_mapping[node_uri]
                neighbors_list = self.intimacy_dictionary[node_id]
                role_ids = [self.wl_role_ids[node_id]]
                position_ids = [i + 1 for i in range(num_of_neighbors + 1)]
                hop_ids = [1]
                for neighbor, intimacy_score in neighbors_list:
                    role_ids.append(self.wl_role_ids[neighbor])
                    if neighbor in self.hop_distance_dict[node_id]:
                        hop_ids.append(self.hop_distance_dict[node_id][neighbor] + 1)
                    else:
                        hop_ids.append(0)
                raw_features_list.append(raw_feature)
                role_ids_list.append(role_ids)
                position_ids_list.append(position_ids)
                hop_ids_list.append(hop_ids)
            else:
                raw_features_list.append([0.0] * number_of_raw_features)
                role_ids_list.append([0] * (num_of_neighbors + 1))
                position_ids_list.append([1 if i == 0 else 0 for i in range(num_of_neighbors + 1)])
                hop_ids_list.append([1 if i == 0 else 0 for i in range(num_of_neighbors + 1)])

        raw_feature_embedding = torch.tensor(raw_features_list, dtype=torch.float32)
        wl_embedding = torch.tensor(role_ids_list, dtype=torch.long)
        hop_embeddings = torch.tensor(hop_ids_list, dtype=torch.long)
        int_embeddings = torch.tensor(position_ids_list, dtype=torch.long)
        return raw_feature_embedding, wl_embedding, hop_embeddings, int_embeddings

    def collate_fn(self, data_list):

        batch_size = len(data_list)
        # Creates a list of all the triples in the form of [head,tail,relation] and splits it across lines
        # in chunks, where each chunk has shape of maximum (batch_size,2)
        head_tail_pairs, relations = torch.stack(data_list).split(2, dim=1)
        relations = relations.to(self.device)
        head_tail_pairs = head_tail_pairs.to(self.device)
        if "foodon" in self.graph_name.lower():
            negative_samples, head_or_tail = _in_batch_negative_sampling(head_tail_pairs, self.num_negative_samples,
                                                                         self.device, self.entities_withoutbns)
            head_or_tail = head_or_tail.view(batch_size, self.num_negative_samples)
            negative_samples = sanitize_negative_samples_mixed(
                negative_samples,
                head_tail_pairs,
                relations,
                self.graph_ids,
                self.entities_withoutbns,
                head_or_tail,
            )
        else:
            negative_samples, head_or_tail = _in_batch_negative_sampling(head_tail_pairs, self.num_negative_samples,
                                                                         self.device)
            head_or_tail = head_or_tail.view(batch_size, self.num_negative_samples)
            negative_samples = sanitize_negative_samples_mixed(
                negative_samples,
                head_tail_pairs,
                relations,
                self.graph_ids,
                self.entities,
                head_or_tail,
            )

        positive_triples_tokens, positive_triples_mask = self.get_entity_description_tokens_masks(
            head_tail_pairs.reshape(-1).flatten())
        raw_features, wl_role_ids, hop_ids, intimacy_ids, graph_mask = self.get_batch_features(
            head_tail_pairs.reshape(-1).flatten())
        negative_triples_tokens, negative_triples_mask = self.get_entity_description_tokens_masks(
            negative_samples.reshape(-1).flatten())
        negative_raw_features, negative_wl_role_ids, negative_hop_ids, negative_intimacy_ids, negative_graph_mask = self.get_batch_features(
            negative_samples.reshape(-1).flatten())
        return (positive_triples_tokens, positive_triples_mask, relations, raw_features,
                wl_role_ids, hop_ids, intimacy_ids, graph_mask, negative_triples_tokens,
                negative_triples_mask, negative_raw_features, negative_wl_role_ids,
                negative_hop_ids, negative_intimacy_ids, negative_graph_mask)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]







