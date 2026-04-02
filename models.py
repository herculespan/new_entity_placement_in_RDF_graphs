import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from utils import BertEncoder, BertEmbeddings, freeze_text_encoder
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertPooler
from config import GraphBertConfig
import torch.nn.functional as F
import transformers
import numpy
import pandas
import scipy
import networkx
print(transformers.__version__)
print(torch.__version__)
print(numpy.__version__)
print(pandas.__version__)
print(scipy.__version__)



quit()






def transe_score(heads, tails, rels):
    return -torch.norm(heads + rels - tails, dim=-1, p=1)

def complex_score(heads, tails, rels):
    heads_real, heads_imag = torch.chunk(heads, 2, dim=-1)
    tails_real, tails_imag = torch.chunk(tails, 2, dim=-1)
    rels_real, rels_imag = torch.chunk(rels, 2, dim=-1)

    return torch.sum(heads_real * rels_real *tails_real
                     + heads_real * rels_imag * tails_imag
                     + heads_imag * rels_real * tails_imag
                     - tails_real * heads_imag * rels_imag, dim=-1)

def l2_regularization(heads, tails, rels):
    reg_loss = 0.0
    for tensor in (heads, tails, rels):
        reg_loss += torch.mean(tensor ** 2)

    return reg_loss / 3.0


def margin_loss(positive_scores, negative_scores):
    loss = 1 - positive_scores + negative_scores
    loss[loss < 0] = 0
    margin_loss = torch.mean(loss)
    return margin_loss

def logistic_softplus_loss(positive_scores, negative_scores):
    positives_loss = F.softplus(-positive_scores).mean()
    negatives_loss = F.softplus(negative_scores).mean()
    return positives_loss + negatives_loss


class TextEncoder(nn.Module):
    def __init__(self, dimension: int, encoder):
        super(TextEncoder, self).__init__()
        self.encoder = encoder

    def forward(self, text_tokens, text_mask):
        embedding_vectors = self.encoder(input_ids=text_tokens, attention_mask=text_mask)
        cls_token = embedding_vectors.last_hidden_state[:, 0, :]
        bert_embeddings = cls_token

        return bert_embeddings


class GraphTopologyEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(GraphTopologyEncoder, self).__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, head_mask=None, residual_h=None):
        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers
        embedding_output = self.embeddings(raw_features=raw_features, wl_role_ids=wl_role_ids,
                                           init_pos_ids=init_pos_ids, hop_dis_ids=hop_dis_ids)
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask, residual_h=residual_h)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return pooled_output

    def run(self):
        pass


class Fusion(nn.Module):
    def __init__(self, topology_features_dimension: int, text_embeddings_dimension: int,model_dimension: int, dropout_p: float):
        super().__init__()
        self.text_dimension = text_embeddings_dimension
        self.topology_dimensions = topology_features_dimension
        self.model_dimension = model_dimension
        self.project_topology = nn.Linear(topology_features_dimension, model_dimension, bias=False)
        nn.init.xavier_uniform_(self.project_topology.weight)

        self.project_text = nn.Linear(self.text_dimension,self.model_dimension, bias=False)
        nn.init.xavier_uniform_(self.project_text.weight)

        # Normalization layers for the embeddings
        self.text_norm_layer = nn.LayerNorm(self.model_dimension, elementwise_affine=False)
        self.graph_norm_layer = nn.LayerNorm(self.model_dimension, elementwise_affine=False)
        self.fused_norm_layer = nn.LayerNorm(self.model_dimension, elementwise_affine=False)

        # Dropout layers for the embeddings
        self.text_dropout = nn.Dropout(dropout_p)
        self.graph_dropout = nn.Dropout(dropout_p)
        self.fused_dropout = nn.Dropout(dropout_p)

        # Create a layer initialized with zero weights, to convert embedding to scaler
        # +1 is used for the [UNK_GRAPH] which is a representation for entities not present
        # in the graph
        self.emb_to_scalar = nn.Linear(self.topology_dimensions + self.model_dimension + 1, 1)
        # Set weights and bias to zero at the beginning of the training
        nn.init.zeros_(self.emb_to_scalar.weight)
        nn.init.constant(self.emb_to_scalar.bias,1.0)

    def forward(self, text_embeddings: torch.Tensor, graph_encoder_output: torch.Tensor, graph_gate: torch.Tensor):
        text_embeddings = self.project_text(text_embeddings)
        text_embedding = self.text_norm_layer(text_embeddings)
        text_embedding = self.text_dropout(text_embedding)

        gate_input = torch.cat([text_embedding, graph_encoder_output, graph_gate], dim=-1)

        a = torch.sigmoid(self.emb_to_scalar(gate_input))
        graph_embedding = self.project_topology(graph_encoder_output)
        graph_embedding = self.graph_norm_layer(graph_embedding)
        graph_embedding = self.graph_dropout(graph_embedding)
        fused_embedding = a * text_embedding + (1 - a) * graph_embedding
        normalized_fused_embedding = self.fused_norm_layer(fused_embedding)
        normalized_fused_embedding = self.fused_dropout(normalized_fused_embedding)

        return normalized_fused_embedding


class InductiveLinkPrediction(nn.Module):
    def __init__(self, embedding_dimension: int,
                 score_fn, loss_fn,
                 regularizer: float,
                 relations_descriptions,
                 dropout_p: float,
                 max_wl_idx: int,
                 mapper_lambda:float=0.1,
                 pred_graph_confidence: float=0.5,
                 mapper_hidden:int=256,
                 modality_dropout_p:float=0.3):
        super().__init__()
        graph_bert_config = GraphBertConfig(output_hidden_states=False, output_attentions=False,max_wl_role_index=max_wl_idx)
        self.dimension = embedding_dimension
        self.score_fn = score_fn
        self.num_of_topology_features = graph_bert_config.x_size
        self.regularizer = regularizer
        self.loss_fn = loss_fn
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_encoder = TextEncoder(self.dimension, self.bert_encoder)
        self.text_dimension = int(self.text_encoder.encoder.config.hidden_size)

        freeze_text_encoder(self.text_encoder, 6, True)
        # # Create the [UNK_GRAPH] representation and initialize it.
        # self.unk_graph = nn.Parameter(torch.zeros(graph_bert_config.hidden_size))
        # nn.init.normal_(self.unk_graph, mean=0, std=0.02)

        #Code block to create the mapper from text to graph
        self.mapper_lambda = mapper_lambda
        self.predicted_graph_confidence = pred_graph_confidence
        self.modality_dropout = modality_dropout_p

        #Module architecture
        self.text_to_graph_module = nn.Sequential(
            nn.Linear(self.text_dimension, mapper_hidden),
            nn.GELU(),
            nn.Linear(mapper_hidden,graph_bert_config.hidden_size)
        )

        #Initialize the mapper layers and set bias to zero
        for layer in self.text_to_graph_module:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


        # Define the BERT model to encode the relations
        self.linear_layer = nn.Linear(self.bert_encoder.config.hidden_size, self.dimension, bias=False)
        # Define a lookup table of size (num_of_relations,model_dimension)
        self.relation_embedding_table = torch.nn.Embedding(relations_descriptions.shape[0], self.dimension)
        self._relations_embedding(relations_descriptions)

        self.graph_encoder = GraphTopologyEncoder(graph_bert_config)
        self.fused_embeddings = Fusion(graph_bert_config.hidden_size, self.text_dimension,self.dimension, dropout_p)

    def _relations_embedding(self, relations_descriptions):
        with torch.no_grad():
            attention_mask = (relations_descriptions > 0).long()

            bert_embedding = self.bert_encoder(input_ids=relations_descriptions, attention_mask=attention_mask)
            cls_token = bert_embedding.last_hidden_state[:, 0, :]
            embedding_vector = self.linear_layer(cls_token)
            self.relation_embedding_table.weight.data.copy_(embedding_vector)

    def compute_loss(self, positive_scores, negative_scores, heads, tails, relations):
        base_loss = self.loss_fn(positive_scores, negative_scores)
        regularization_loss = self.regularizer * l2_regularization(heads, tails, relations)
        return base_loss + regularization_loss

    def compute_embeddings(self, text_tokens, text_mask, raw_features, wl_roles, hop_ids, intimacy_positions,
                           graph_mask,return_mapper_loss:bool=False):

        real_graph_embeddings = graph_mask.bool()
        #Text embeddings
        text_embeddings = self.text_encoder(text_tokens, text_mask)
        #Graph encoder output
        graph_encoder_output = self.graph_encoder.forward(raw_features, wl_roles, intimacy_positions, hop_ids)

        #Predicted graph encoder output
        predicted_graph_output = self.text_to_graph_module(text_embeddings)

        # Give the [UNK_GRAPH] representation the correct shape in order to replace
        # every entity in graph embeddings with this representation
        # unk = self.unk_graph.unsqueeze(0).expand_as(graph_encoder_output)

        if self.training and self.modality_dropout > 0.0:
            drop = (torch.rand_like(graph_mask.float()) < self.modality_dropout) & real_graph_embeddings

        else:
            drop = torch.zeros_like(real_graph_embeddings, dtype = torch.bool)

        real_graph_embeddings = real_graph_embeddings & (~drop)

        graph_embeddings = torch.where(real_graph_embeddings, graph_encoder_output, predicted_graph_output)


        gate_hint = torch.where(real_graph_embeddings,
                                torch.ones_like(graph_mask,dtype = text_embeddings.dtype),
                                torch.full_like(graph_mask,float(self.predicted_graph_confidence),dtype=text_embeddings.dtype))

        fused_embeddings = self.fused_embeddings.forward(text_embeddings, graph_embeddings, gate_hint)

        if not return_mapper_loss:
            return fused_embeddings

        mask = graph_mask.bool().squeeze(1)
        if mask.any():
            predicted_graph_embeddings = F.normalize(predicted_graph_output[mask], p=2, dim=-1)
            target_graph_embeddings = F.normalize(graph_encoder_output.detach()[mask], p=2, dim=-1)
            text_to_graph_module_loss = F.mse_loss(predicted_graph_embeddings,target_graph_embeddings)
        else:
            text_to_graph_module_loss = torch.tensor(0.0,device=text_embeddings.device,dtype = text_embeddings.dtype)

        return fused_embeddings, text_to_graph_module_loss



    def forward(self,
                text_tokens,
                text_mask,
                relation_ids,
                raw_features, wl, hop, int_emb,
                graph_mask,
                negative_samples_tokens,
                negative_samples_mask,
                neg_raw_features, neg_wl, neg_hop, neg_int_emb,
                neg_graph_mask):
        batch_size = relation_ids.size(0)
        number_of_negative_samples = int(negative_samples_tokens.size(0) / (2 * batch_size))

        relation_embeddings = self.relation_embedding_table(relation_ids)

        entity_embeddings,mapper_loss = self.compute_embeddings(text_tokens, text_mask, raw_features, wl, hop, int_emb, graph_mask,True)

        # print("Entity embeddings shape:", entity_embeddings.shape)

        entity_embeddings = entity_embeddings.view(batch_size, 2, self.dimension)
        heads, tails = entity_embeddings[:, 0, :], entity_embeddings[:, 1, :]
        # print(heads.shape)
        # print(tails.shape)
        positive_scores = self.score_fn(heads, tails, relation_embeddings.squeeze())
        # print("Positive scores",positive_scores)
        # print("Positive scores shape",positive_scores.shape)

        negative_fuse_embeddings = self.compute_embeddings(negative_samples_tokens, negative_samples_mask,
                                                           neg_raw_features, neg_wl, neg_hop, neg_int_emb,
                                                           neg_graph_mask)
        neg_embeddings = negative_fuse_embeddings.view((batch_size, int(number_of_negative_samples), 2, self.dimension))
        # entities_in_negative_embeddings = negative_fuse_embeddings.size(0)
        # per_triple_entities = entities_in_negative_embeddings // batch_size
        # assert per_triple_entities % 2 == 0

        # # negatives_per_triple = per_triple_entities // 2
        # # print("Negative entity embeddings shape", negative_fuse_embeddings.shape)
        # neg_embeddings = negative_fuse_embeddings.view(batch_size,per_triple_entities,self.dimension)

        # negative_heads, negative_tails = neg_embeddings.chunk(2, dim=1)

        negative_heads = neg_embeddings[..., 0, :]
        negative_tails = neg_embeddings[..., 1, :]
        negative_scores = self.score_fn(negative_heads, negative_tails, relation_embeddings)
        # print("Negative scores",negative_scores)
        # print("Negative scores shape",negative_scores.shape)

        loss = self.compute_loss(positive_scores.unsqueeze(1), negative_scores, heads, tails, relation_embeddings)
        model_loss = loss + (self.mapper_lambda * mapper_loss)
        return model_loss
