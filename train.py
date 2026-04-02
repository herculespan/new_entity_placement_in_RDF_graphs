import networkx as nx
from torch.utils.data import DataLoader
import os
from data import TextWithGraphTopologyDataset
from utils import select_device, get_logger, id_to_row, compute_ranking, head_tail_filters
import data
import models
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch
import itertools
from torch.amp import autocast, GradScaler
import numpy as np
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import utils
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler



def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


MIN_DELTA = 0.001

PATIENCE = 5

NUMBER_OF_NEGATIVE_SAMPLES = 32

NUMBER_OF_EPOCHS = 40

BATCH_SIZE = 32

EVAL_BATCH_SIZE = 64

EMB_BATCH_SIZE = 512

TEXT_LEARNING_RATE = 2e-5

GRAPH_FUSION_LEARNING_RATE = 5e-4

RELATION_EMBEDDING_LEARNING_RATE = 5e-4

REGULARIZER_WEIGHT = 1e-5

# FUSION_LEARNING_RATE = 5e-4

TEXT_WEIGHT_DECAY = 0.01

WEIGHT_DECAY = 1e-4

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
# NUMBER_OF_TOPOLOGY_FEATURES = 80

if torch.cuda.is_available():
    DEVICE = torch.device("cuda", get_local_rank())
else:
    DEVICE = select_device()

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

K_VALUES = [1, 3, 10]

OUTPUT_PATH = "models/"


def train(training_parameters_dict, hyperparameters_dict, logger, hparams_combo_index):
    regularizer_weight = hyperparameters_dict.get('regularizer_weight', REGULARIZER_WEIGHT)
    epochs = hyperparameters_dict.get('epochs', NUMBER_OF_EPOCHS)
    graph_fusion_lr = hyperparameters_dict.get('fusion_learning_rate', GRAPH_FUSION_LEARNING_RATE)
    relations_lr = hyperparameters_dict.get("relation_learning_rate", RELATION_EMBEDDING_LEARNING_RATE)


    train_data = training_parameters_dict["train_data"]
    train_loader = training_parameters_dict["train_loader"]
    wl_role_index = train_data.wl_role_indices.max().item() + 1
    logger.info(f"Train for {epochs} epochs")

    model = models.InductiveLinkPrediction(192,
                                           models.transe_score,
                                           models.margin_loss,
                                           regularizer_weight,
                                           train_data.relation_description,
                                           0.15,
                                           wl_role_index).to(DEVICE)

    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
            find_unused_parameters=True)

    base_model = unwrap_model(model)

    # Split the parameters in groups in order to have a specific learning rate for every model (BERT, GRAPH-BERT, Fusion & relation embeddings)
    text_encoder_named_parameters = [(name, param) for (name, param) in base_model.text_encoder.named_parameters() if
                                     param.requires_grad]
    graph_encoder_named_parameters = list(base_model.graph_encoder.named_parameters())

    text_encoder_params, graph_encoder_params, fusion_params = utils.split_params_for_lrs(base_model)

    text_encoder_ids = {id(param) for param in text_encoder_params}
    graph_encoder_ids = {id(param) for param in graph_encoder_params}
    fusion_named_parameters = [(name, param) for (name, param) in base_model.named_parameters() if
                               (id(param) not in text_encoder_ids) and (id(param) not in graph_encoder_ids) and (
                                   not name.startswith("relation_embedding_table")) and (
                                   not name.startswith("relation_projection")) and (
                                   not name.startswith("relations_gate"))]
    relation_embeddings_parameters = [(name, param) for (name, param) in base_model.named_parameters() if
                                      (name.startswith("relation_embedding_table")) or (
                                          name.startswith("relation_projection")) or (
                                          name.startswith("relations_gate"))]

    optimizer_params = []
    optimizer_params += utils.create_params_groups(text_encoder_named_parameters, TEXT_LEARNING_RATE, TEXT_WEIGHT_DECAY)
    optimizer_params += utils.create_params_groups(graph_encoder_named_parameters, graph_fusion_lr, WEIGHT_DECAY)
    optimizer_params += utils.create_params_groups(fusion_named_parameters, graph_fusion_lr, WEIGHT_DECAY)
    optimizer_params += utils.create_params_groups(relation_embeddings_parameters, relations_lr, WEIGHT_DECAY)

    logger.info(f"#text params: {len(text_encoder_named_parameters)}")
    logger.info(f"#graph params: {len(graph_encoder_named_parameters)}")
    logger.info(f"#fusion params: {len(fusion_named_parameters)}")
    logger.info(f"#relation params: {len(relation_embeddings_parameters)}")
    logger.info(f"#fusion learning rate: {graph_fusion_lr}")
    logger.info(f"#reltions learning rate: {relations_lr}")
    logger.info(f"#regularizer weight: {regularizer_weight}")
    logger.info(f"save name: {hparams_combo_index}")

    # Initialize the optimizer with a scheduler
    optimizer = AdamW(optimizer_params)
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * steps)

    def warmup_lambda(step: int):
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    initial_lrs = [group['lr'] for group in optimizer.param_groups]
    min_lrs = [max(lr * 0.01, 1e-7) for lr in initial_lrs]
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode="max",
                                  factor=0.5,
                                  patience=3,
                                  threshold=1e-3,
                                  threshold_mode="abs",
                                  cooldown=0,
                                  min_lr=min_lrs)
    best_mrr = float('-inf')
    epochs_without_improvement = 0
    use_amp = (DEVICE.type == "cuda")
    logger.info(f"Is use amp?: {use_amp}")
    scaler = GradScaler(enabled=use_amp)
    best_model_state = None

    max_confidence = 0.4
    global_step = 0
    for epoch in range(epochs):

        if is_distributed() and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        base_model.modality_dropout = modality_dropout_scheduler(epoch)
        base_model.predicted_graph_confidence = predicted_confidence_scheduler(epoch, max_confidence=max_confidence)
        logger.info("==================================")
        logger.info(f"Starting training epoch {epoch + 1}")
        logger.info(f"Modality dropout rate: {base_model.modality_dropout:.2f}")
        logger.info(f"Predicted graph confidence: {base_model.predicted_graph_confidence:.2f}")
        logger.info("==================================")
        train_loss = 0
        for batch_index, batch in enumerate(train_loader):
            batch = [tensor.to(DEVICE, non_blocking=True) for tensor in batch]

            optimizer.zero_grad(set_to_none=True)

            scale_before = scaler.get_scale() if use_amp else None
            with autocast(device_type=DEVICE.type):
                loss = model(*batch)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)  # Gradients are in their real scale
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping to avoid spikes

            scaler.step(optimizer)
            scaler.update()

            scale_after = scaler.get_scale() if use_amp else None

            skipped_step = scale_after < scale_before

            if not skipped_step and global_step < warmup_steps:
                warmup_scheduler.step()
                global_step += 1
            train_loss += loss.item()

            if batch_index % 1000 == 0:
                logger.info(f"Train loss for batch {batch_index}: {train_loss / (batch_index + 1):.3f}")

        logger.info(f"Training loss for epoch {epoch + 1}: {train_loss / len(train_loader):.3f}")
        if is_main_process():
            logger.info("Raw evaluation on a sample of the train dataset")
            train_evaluation_mrr = evaluate(training_parameters_dict['train_evaluation_loader'],
                                            model,
                                            training_parameters_dict['train_data'],
                                            training_parameters_dict["train_entities"],
                                            logger,
                                            len(training_parameters_dict['validation_loader']))

            logger.info(f"Raw evaluation MRR on a sample of the train set: {train_evaluation_mrr:.3f}")
            # Add the raw evaluation with the valid set here
            logger.info("Raw evaluation on the validation dataset")
            validation_mrr = evaluate(training_parameters_dict['validation_loader'],
                                      model,
                                      training_parameters_dict['train_data'],
                                      training_parameters_dict['train_validation_entities'],
                                      logger,
                                      len(training_parameters_dict['validation_loader']))
            logger.info(f"Raw evaluation MRR on the validation set: {validation_mrr:.3f}")

        else:
            train_evaluation_mrr = 0.0
            validation_mrr = 0.0

        validation_mrr = broadcast_float(validation_mrr, DEVICE)
        scheduler.step(validation_mrr)

        # IMPLEMENT EARLY STOPPING SCENARIO
        if validation_mrr >= best_mrr + MIN_DELTA:
            best_mrr = validation_mrr
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(unwrap_model(model).state_dict())
            torch.save(best_model_state, f"{OUTPUT_PATH}/best_model_{hparams_combo_index}.pt")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement > PATIENCE and is_main_process():
            logger.info("Early stopping to avoid overfitting")

            break
        torch.cuda.empty_cache()
        if is_main_process():
            logger.info(f"Best MRR value: {best_mrr:.3f} ")

    logger.info(f"Completed training process at epoch {epoch + 1} out of {NUMBER_OF_EPOCHS}")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return best_mrr, model


@torch.no_grad()
def evaluate(triples_to_evaluate,
             model,
             texts,
             entities_tensor,
             logger,
             max_number_of_batches,
             number_of_inductive_triples=0,
             hierarchical_relations=None,
             filtering_graph=None,
             inductive_entities=None):
    # Define number of entities and initiate the necessary variables.
    # Set the model in evaluation mode
    model.eval()
    num_entities = entities_tensor.shape[0]
    entities_embeddings = torch.zeros(num_entities, unwrap_model(model).dimension, device=DEVICE)
    idx = 0
    is_filtered_evaluation = filtering_graph is not None
    max_entity_ids = int(torch.max(entities_tensor))
    raw_id_to_embedding_row = id_to_row(entities_tensor, max_entity_ids + 1) if is_filtered_evaluation else id_to_row(
        entities_tensor)
    k_values = torch.tensor(K_VALUES, device=DEVICE)
    # Initialize the metrics for evaluating the overall predictive capability of the model
    raw_mrr = torch.zeros((), device=DEVICE)
    raw_hits = torch.zeros(len(K_VALUES, ), device=DEVICE)
    hits = torch.zeros(len(K_VALUES), device=DEVICE)
    raw_hits_at_k = {k: 0.0 for k in K_VALUES}
    filtered_mrr = 0.0
    filtered_hits_at_k = {k: 0.0 for k in K_VALUES}
    number_of_predictions = 0

    found_hierarchy = torch.zeros((max_entity_ids + 1), device=DEVICE, dtype=torch.int32)
    found_in_head_hierarchy = torch.zeros((max_entity_ids + 1), device=DEVICE, dtype=torch.int32)
    found_in_tail_hierarchy = torch.zeros((max_entity_ids + 1), device=DEVICE, dtype=torch.int32)
    found_neighborhood_hierarchical = torch.zeros((max_entity_ids + 1, 2),
                                                  device=DEVICE, dtype=torch.int32)
    found_neighborhood_nonhierarchical = torch.zeros(
        (max_entity_ids + 1, 2),
        device=DEVICE, dtype=torch.int32)
    wrong_placements = torch.zeros((max_entity_ids + 1), device=DEVICE, dtype=torch.int32)
    wrong_head_placements = torch.zeros((max_entity_ids + 1), device=DEVICE, dtype=torch.int32)
    wrong_head_placements_nh = torch.zeros((max_entity_ids + 1), device=DEVICE, dtype=torch.int32)
    wrong_tail_placements = torch.zeros((max_entity_ids + 1), device=DEVICE, dtype=torch.int32)
    wrong_tail_placements_nh = torch.zeros((max_entity_ids + 1), device=DEVICE, dtype=torch.int32)

    # Initialize the metrics for model's inductive link prediction capability (usage of the average RRs)
    filtered_mrr_per_bucket = torch.zeros((3, 1), device=DEVICE)

    items_in_bucket = torch.zeros((3, 1), device=DEVICE)

    # Initialize the metrics for model's performance in identifying the position of the new concept (retrieval task)
    direction_reciprocal_ranks = torch.full((number_of_inductive_triples, 4), float("nan"), device=DEVICE)
    # direction_ranks = torch.full((number_of_inductive_triples,3),float("nan"),device=DEVICE)
    # items_in_bucket = torch.zeros((number_of_inductive_triples,3),device=DEVICE)

    # Compute the embeddings of entities with the latest model's weights
    while idx < num_entities:
        batch_ents = entities_tensor[idx:idx + EMB_BATCH_SIZE]
        # Check the function in data.py since the ids are not necessary (is batch_ents needed?)
        text_tokens, text_masks = texts.get_entity_description_tokens_masks(batch_ents)
        text_tokens, text_masks = text_tokens.to(DEVICE), text_masks.to(DEVICE)
        # text_tokens, text_masks = text_tokens[batch_ents].to(DEVICE), text_masks[batch_ents].to(DEVICE)
        # raw_features,wl_role,hop_ids,intimacy_ids = texts.extract_batch_features(batch_ents)
        raw_features, wl_role, hop_ids, intimacy_ids, graph_mask = texts.get_batch_features(batch_ents)
        raw_features, wl_role, hop_ids, intimacy_ids, graph_mask = raw_features.to(DEVICE), wl_role.to(
            DEVICE), hop_ids.to(DEVICE), intimacy_ids.to(DEVICE), graph_mask.to(DEVICE)
        batch_embeddings = model.compute_embeddings(text_tokens, text_masks, raw_features, wl_role, hop_ids,
                                                    intimacy_ids, graph_mask)
        entities_embeddings[idx:idx + batch_embeddings.shape[0]] = batch_embeddings
        idx += EMB_BATCH_SIZE

    unsqueezed_entities_embeddings = entities_embeddings.unsqueeze(0)
    idx = 0

    # Begin the evaluation process
    for batch_index, batch_triples in enumerate(triples_to_evaluate):
        if batch_index == max_number_of_batches:
            break

        # Get heads, tails embeddings and create the true entities tensor
        heads, tails, relations = torch.chunk(batch_triples, chunks=3, dim=1)
        heads = raw_id_to_embedding_row[heads]
        tails = raw_id_to_embedding_row[tails]

        # Ensure that all the heads are present in the graph
        assert (heads >= 0).all()
        assert (tails >= 0).all()

        true_entities = torch.cat((heads, tails)).to(DEVICE)
        # Get the head, tail and relation embeddings
        head_embeddings = entities_embeddings[heads]
        tail_embeddings = entities_embeddings[tails]
        relations_embeddings = model.relation_embedding_table(relations.to(DEVICE))
        # relations_embeddings = model.compute_relation_embeddings(relations.squeeze(-1).to(DEVICE))
        # relations_embeddings = relations_embeddings.unsqueeze(1)
        # Score on heads and tails by replacing either head or tail with every candidate entity
        head_predictions = model.score_fn(unsqueezed_entities_embeddings, tail_embeddings, relations_embeddings)
        tail_predictions = model.score_fn(head_embeddings, unsqueezed_entities_embeddings, relations_embeddings)

        predictions = torch.cat((head_predictions, tail_predictions)).to(DEVICE)
        number_of_predictions += predictions.shape[0]

        # Same process but for filtered graph
        if is_filtered_evaluation:
            # Filter the graph to score lowest the other true triples
            heads_mask, tails_mask = head_tail_filters(batch_triples, num_entities, filtering_graph,
                                                       raw_id_to_embedding_row)
            graph_masks = torch.cat((heads_mask, tails_mask)).to(DEVICE)
            # Place the other true triples in the last positions of the predictions tensor
            predictions[graph_masks] = predictions.min() - 1.0

            # Compute filtered performance metrics for the overall predictive capability of the model
            filtered_ranking = compute_ranking(predictions, true_entities)
            filtered_reciprocal_ranking = filtered_ranking.reciprocal()
            filtered_mrr += filtered_reciprocal_ranking.sum().item()
            hits += (filtered_ranking <= k_values).sum(dim=0)
            # for index,k in enumerate(K_VALUES):
            #     filtered_hits_at_k[k] += hits[index].item()

            if inductive_entities.any():
                number_of_batch_inductive_triples = utils.compute_number_of_inductive_triples(batch_triples,
                                                                                              inductive_entities)
                batch_reciprocals, batch_counts = utils.split_in_buckets(batch_triples, inductive_entities,
                                                                         filtered_reciprocal_ranking,
                                                                         number_of_batch_inductive_triples, True)
                filtered_mrr_per_bucket += batch_reciprocals
                items_in_bucket += batch_counts
                # for index,k in enumerate(K_VALUES):
                #     filtered_hits_at_k_per_bucket[index][k] += hits[index].item()

                batch_direction_reciprocal_ranks = utils.split_in_buckets(batch_triples, inductive_entities,
                                                                          filtered_reciprocal_ranking,
                                                                          number_of_batch_inductive_triples, False)
                direction_reciprocal_ranks[
                    idx:idx + batch_direction_reciprocal_ranks.shape[0]] = batch_direction_reciprocal_ranks
                idx += batch_direction_reciprocal_ranks.shape[0]

                (num_wrong_placements, num_head_wrong_placements, num_head_wrong_placements_nh,
                 num_tail_wrong_placements,
                 num_tail_wrong_placements_nh, in_hierarchy, in_head_hierarchy, in_tail_hierarchy,
                 in_neighborhood_hierarchical, in_neighborhood_nh) = utils.evaluate_wrong_replacements(batch_triples,
                                                                                                       inductive_entities,
                                                                                                       true_entities,
                                                                                                       predictions,
                                                                                                       hierarchical_relations,
                                                                                                       raw_id_to_embedding_row,
                                                                                                       texts.ids2entities,
                                                                                                       texts.hierarchy,
                                                                                                       texts.neighborhood)
                found_hierarchy += in_hierarchy
                found_in_head_hierarchy += in_head_hierarchy
                found_in_tail_hierarchy += in_tail_hierarchy
                found_neighborhood_hierarchical += in_neighborhood_hierarchical
                found_neighborhood_nonhierarchical += in_neighborhood_nh
                wrong_placements += num_wrong_placements
                wrong_head_placements += num_head_wrong_placements
                wrong_head_placements_nh += num_head_wrong_placements_nh
                wrong_tail_placements += num_tail_wrong_placements
                wrong_tail_placements_nh += num_tail_wrong_placements_nh




        else:
            # Compute raw evaluation metrics (not filtered)
            raw_ranking = compute_ranking(predictions, true_entities)
            raw_mrr += raw_ranking.reciprocal().sum()
            raw_hits += (raw_ranking <= k_values).sum(dim=0)
            # for index,k in enumerate(K_VALUES):
            #     raw_hits_at_k[k] += raw_hits[index].item()

    if is_filtered_evaluation:
        filtered_mrr /= number_of_predictions
        filtered_hits = (hits / number_of_predictions).float().cpu().tolist()
        # logger.info(f"Filtered evaluation mrr: {filtered_mrr:.3f}")
        for index, k in enumerate(filtered_hits_at_k.keys()):
            filtered_hits_at_k[k] = np.round(filtered_hits[index], 3)
        logger.info(f"Filtered evaluation hits at k: {filtered_hits_at_k}")
        logger.info(f"Filtered evaluation MRR: {filtered_mrr:.3f}")
        predictions_in_bucket = 2 * items_in_bucket
        filtered_mrr_per_bucket /= predictions_in_bucket
        logger.info("Filtered MRR per bucket: {filtered_mrr_per_bucket}")
        logger.info(f"Filtered MRR in triples with new head: {filtered_mrr_per_bucket[0].item():.2f}")
        logger.info(f"Filtered MRR in triples with tail head: {filtered_mrr_per_bucket[2].item():.2f}")
        logger.info(f"Filtered MRR in triples with new head and tail: {filtered_mrr_per_bucket[2].item():.2f}")
        macro_mrr_per_bucket, macro_mrr_per_rel_category = utils.compute_macro_mrrs(direction_reciprocal_ranks,
                                                                                    hierarchical_relations)
        logger.info(
            f"""MRR per relation type for bucket with head as inductive entities {macro_mrr_per_bucket[0].item():.2f}
                \nMRR per relation type for bucket with tail as inductive entities {macro_mrr_per_bucket[1].item():.2f}
                \nMRR per relation type for bucket with both head and tail as inductive entities {macro_mrr_per_bucket[2].item():.2f}\n""")
        logger.info("=====HIERARCHICAL RELATIONS=====")
        logger.info(
            f"""MRR per relation category for bucket with head as inductive entities {macro_mrr_per_rel_category[1, 0].item():.2f} 
                        \nMRR per relation category for bucket with tail as inductive entities {macro_mrr_per_rel_category[1, 1].item():.2f}
                        \nMRR per relation category for bucket with both head and tail as inductive entities {macro_mrr_per_rel_category[1, 2].item():.2f}\n""")
        logger.info("=====NON HIERARCHICAL RELATIONS=====")
        logger.info(
            f"""MRR per relation category for bucket with head as inductive entities {macro_mrr_per_rel_category[0, 0].item():.2f} 
                    \nMRR per relation category for bucket with tail as inductive entities {macro_mrr_per_rel_category[0, 1].item():.2f}
                    \nMRR per relation category for bucket with both head and tail as inductive entities {macro_mrr_per_rel_category[0, 2].item():.2f}\n""")

        percentage_in_hierarchy = torch.sum(found_hierarchy) * 100 / torch.sum(wrong_placements)
        logger.info(f"""Rate of wrong entity predictions in the correct hierarchy: {percentage_in_hierarchy:.2f}%""")
        den = (torch.sum(wrong_head_placements) + torch.sum(wrong_head_placements_nh)).float()
        if den == 0.0:
            head_percentage_in_hierarchy = torch.tensor(0.0)
        else:
            head_percentage_in_hierarchy = torch.sum(found_in_head_hierarchy) * 100 / den
        logger.info(
            f"""Rate of wrong head entity predictions in the correct hierarchy: {head_percentage_in_hierarchy:.2f}%""")
        den = (torch.sum(wrong_tail_placements) + torch.sum(wrong_tail_placements_nh)).float()
        if den == 0.0:
            tail_percentage_in_hierarchy = torch.tensor(0.0)
        else:
            tail_percentage_in_hierarchy = torch.sum(found_in_tail_hierarchy) * 100 / den
        logger.info(
            f"""Rate of wrong tail entity predictions in the correct hierarchy: {tail_percentage_in_hierarchy:.2f}%""")

        # #Hierarchical relations
        logger.info(f"""============HIERARCHICAL RELATIONS============""")
        den = torch.sum(wrong_head_placements)
        if den == 0.0:
            percentage_heads_in_neighborhood = torch.tensor(0.0)
        else:
            percentage_heads_in_neighborhood = torch.sum(found_neighborhood_hierarchical[:, 0]) * 100 / den
        logger.info(
            f"""Rate of wrong head entity predictions in the correct neighborhood: {percentage_heads_in_neighborhood:.2f}% """)
        den = torch.sum(wrong_tail_placements)
        if den == 0.0:
            percentage_tails_in_neighborhood = torch.tensor(0.0)
        else:
            percentage_tails_in_neighborhood = torch.sum(found_neighborhood_hierarchical[:, 1]) * 100 / den
        logger.info(
            f"""Rate of wrong tail entity predictions in the correct neighborhood: {percentage_tails_in_neighborhood:.2f}% """)
        # #Non-hierarchical relations
        logger.info(f"""============NON HIERARCHICAL RELATIONS============""")
        den = torch.sum(wrong_head_placements_nh)
        if den == 0.0:
            percentage_heads_in_neighborhood_nh = torch.tensor(0.0)
        else:
            percentage_heads_in_neighborhood_nh = torch.sum(found_neighborhood_nonhierarchical[:, 0]) * 100 / den
        logger.info(
            f"""Rate of wrong head entity predictions in the correct neighborhood: {percentage_heads_in_neighborhood_nh:.2f}% """)
        den = torch.sum(wrong_tail_placements_nh)
        if den == 0.0:
            percentage_tails_in_neighborhood_nh = torch.tensor(0.0)
        else:
            percentage_tails_in_neighborhood_nh = torch.sum(found_neighborhood_nonhierarchical[:, 1]) * 100 / den
        logger.info(
            f"""Rate of wrong tail entity predictions in the correct neighborhood: {percentage_tails_in_neighborhood_nh:.2f}% """)
        return filtered_mrr


    else:
        raw_mrr = (raw_mrr / number_of_predictions).float().cpu().item()
        # logger.info(f"Raw evaluation mrr: {raw_mrr}")
        raw_hits = (raw_hits / number_of_predictions).float().cpu().tolist()
        for index, k in enumerate(raw_hits_at_k.keys()):
            raw_hits_at_k[k] = np.round(raw_hits[index], 3)
        logger.info(f"Raw evaluation Hits at k: {raw_hits_at_k}")
        return raw_mrr


def grid_search(training_parameters, hparameters_dict, logger, path_to_triples):
    # Separate the keys from values of the hyperparameter dict and store them into two tuples
    keys, values = zip(*hparameters_dict.items())
    # Initialize the dictionary to return the best model
    best_combination = {"mrr": -1.0, "model": None, "params": None}
    # Take every combination from the cartesian product of the values tuple
    # and use it as training parameters
    for index, hparm_combo in enumerate(itertools.product(*values)):
        if hparm_combo[-2] != hparm_combo[-1]:
            continue

        batch_size = hparm_combo[-2]
        negative_samples = hparm_combo[-1]
        if batch_size == 64 or negative_samples == 64:
            continue
        per_gpu_batch = max(1, batch_size // WORLD_SIZE)
        train_sampler = None
        shuffle = True

        train_data = data.TextWithGraphTopologyDataset(f"{path_to_triples}/train.txt",
                                                       "foodon",
                                                       training_parameters['graph'],
                                                       TOKENIZER,
                                                       "scientific.txt",
                                                       False,
                                                       True,
                                                       negative_samples)

        if is_distributed():
            train_sampler = DistributedSampler(train_data, shuffle=shuffle, drop_last=True)
            shuffle = False

        training_parameters["train_data"] = train_data
        training_parameters["train_entities"] = train_data.entities
        train_loader = DataLoader(train_data,
                                  batch_size=per_gpu_batch,
                                  shuffle=shuffle,
                                  sampler=train_sampler,
                                  collate_fn=train_data.collate_fn,
                                  num_workers=4, pin_memory=True, drop_last=True)
        training_parameters["train_loader"] = train_loader
        train_evaluation_loader = DataLoader(train_data, batch_size=EVAL_BATCH_SIZE, pin_memory=True)
        training_parameters["train_evaluation_loader"] = train_evaluation_loader
        hyperparameters = dict(zip(keys, hparm_combo))
        if is_main_process():
            logger.info(f"Is global negative sampling?: {train_data.is_global_negative_sampling}")
            logger.info(f"{WORLD_SIZE} available GPUs")
            logger.info(f"{per_gpu_batch} batches per GPU")
            logger.info(
                f"Created train data with {negative_samples} negative samples and dataloader with batch size {batch_size}")
            logger.info(f"========HYPERPARAMETER COMBINATION========\n{hyperparameters}")
        mrr, model = train(training_parameters, hyperparameters, logger, index)

        if mrr > best_combination["mrr"]:
            best_combination['mrr'] = mrr
            best_combination['model'] = model
            best_combination['params'] = hyperparameters
            # best_model_state = copy.deepcopy(model.state_dict())
            logger.info(f"Current best model hyperparameters\n{hyperparameters}")
            torch.save(unwrap_model(model).state_dict(), f"{OUTPUT_PATH}/best_global_model.pt")

    if is_main_process():
        logger.info("========BEST HYPERPARAMTER COMBINATION========")
        for parameter_key in best_combination['params'].keys():
            logger.info(f"{parameter_key}: {best_combination['params'][parameter_key]}")

    return best_combination['model']


def main(path_to_triples: str, graph):
    # Initialize the logger of the training process
    train_logger = get_logger()
    train_logger.info("=============================")
    train_logger.info("Starting training process....")
    # Create the dataset objects and the dataloader objects
    train_data = data.TextWithGraphTopologyDataset(f"{path_to_triples}/train.txt",
                                                   "foodon",
                                                   graph,
                                                   TOKENIZER,
                                                   "scientific.txt",
                                                   False)

    valid_data = TextWithGraphTopologyDataset(f"{path_to_triples}/validation.txt",
                                              "foodon",
                                              graph,
                                              TOKENIZER,
                                              "scientific.txt",
                                              False,
                                              )
    valid_loader = DataLoader(valid_data, batch_size=EVAL_BATCH_SIZE)
    if is_main_process():
        train_logger.info("Created validation data dataset and loader")
    test_data = TextWithGraphTopologyDataset(f"{path_to_triples}/test.txt",
                                             "foodon",
                                             graph,
                                             TOKENIZER,
                                             "scientific.txt",
                                             False,
                                             )
    test_loader = DataLoader(test_data, batch_size=EVAL_BATCH_SIZE)
    if is_main_process():
        train_logger.info("Created test data dataset and loader")
    # Define hierarchical relation ids of the graph
    hierarchical_relations = list(
        set(train_data.hierarchical_relations).union(set(valid_data.hierarchical_relations)).union(
            set(test_data.hierarchical_relations)))

    # Code block for the creation of entity sets as part of assembling the graph
    # for filtered metrics computation

    train_entities = train_data.entities
    validation_entities = valid_data.entities
    test_entities = test_data.entities
    train_validation_entities = list(set(train_entities.tolist()).union(set(validation_entities.tolist())))
    train_validation_test_entities = list(set(train_validation_entities).union(set(test_entities.tolist())))
    validation_new_entities = list(set(train_validation_entities).difference(set(train_entities.tolist())))
    test_new_entities = list(set(train_validation_test_entities).difference(set(train_validation_entities)))

    train_validation_triples = torch.cat([train_data.triples, valid_data.triples])
    validation_graph_for_filtering = nx.DiGraph()
    validation_graph_for_filtering.add_weighted_edges_from(train_validation_triples.tolist())

    train_validation_test_triples = torch.cat((train_data.triples, valid_data.triples, test_data.triples))
    test_graph_for_filtering = nx.DiGraph()
    test_graph_for_filtering.add_weighted_edges_from(train_validation_test_triples.tolist())

    # Make the lists tensors
    train_validation_entities = torch.tensor(train_validation_entities, dtype=torch.long)
    train_validation_test_entities = torch.tensor(train_validation_test_entities, dtype=torch.long)
    number_of_val_inductive_triples = utils.compute_number_of_inductive_triples(valid_data.triples,
                                                                                validation_new_entities)
    train_parameters = {
        "validation_loader": valid_loader,
        "train_validation_entities": train_validation_entities,
        "graph": graph,
        # "filtering_graph": validation_graph_for_filtering,
        # "number_of_val_inductive_triples": number_of_val_inductive_triples,
        # "hierarchical_relations": hierarchical_relations,
        # "inductive_entities": validation_new_entities

    }
    hyperparameters = {

        "fusion_learning_rate": [5e-4, 8e-4],
        "regularizer_weight": [1e-5, 1e-6],
        "batch_size": [32, 64],
        "number_of_negative_samples": [32, 64]
    }

    # GRID SEARCH TRAIN
    model = grid_search(train_parameters, hyperparameters, train_logger, path_to_triples)

    # need to check if the model final evaluation must be done the following way
    # state = torch.load(best_model_path,map_location = DEVICE)
    # model = models.InductiveLinkPrediction(192,
    #                                        models.transe_score,
    #                                        models.margin_loss,
    #                                        5e-4,
    #                                        train_data.relation_description,
    #                                        0.15,
    #                                        pred_graph_confidence=0.4).to(DEVICE)
    # model.load_state_dict(state)

    number_of_val_inductive_triples = utils.compute_number_of_inductive_triples(valid_data.triples,
                                                                                validation_new_entities)
    train_logger.info("Filtered evaluation on validation dataset")
    filtered_validation_mrr = evaluate(valid_loader, model, train_data, train_validation_entities, train_logger,
                                       len(valid_loader), number_of_val_inductive_triples, hierarchical_relations,
                                       validation_graph_for_filtering, validation_new_entities)

    train_logger.info("Filtered evaluation on validation dataset MRR: {:.4f}".format(filtered_validation_mrr))
    train_logger.info("Filtered evaluation on test dataset")
    number_of_test_inductive_triples = utils.compute_number_of_inductive_triples(test_data.triples,
                                                                                 test_new_entities)
    filtered_test_mrr = evaluate(test_loader, model, train_data, train_validation_test_entities, train_logger,
                                 len(test_loader), number_of_test_inductive_triples, hierarchical_relations,
                                 test_graph_for_filtering, test_new_entities)
    train_logger.info("Filtered evaluation on test dataset MRR: {:.4f}".format(filtered_test_mrr))


def train_best_combination(path_to_triples, graph_name, hparameters_combo, graph):
    train_logger = get_logger()
    training_parameters = {}
    batch_size = hparameters_combo.get("batch_size", BATCH_SIZE)
    number_of_negative_samples = hparameters_combo.get("number_of_negative_samples", NUMBER_OF_NEGATIVE_SAMPLES)
    train_logger.info("=============================")
    train_logger.info("Starting training process....")
    train_logger.info(f"Batch size {batch_size}")
    train_logger.info(f"Number of negative samples {number_of_negative_samples}")
    # --- Training dataset (with negative sampling) ---
    train_data = data.TextWithGraphTopologyDataset(
        f"{path_to_triples}/train.txt",
        graph_name, graph, TOKENIZER, "scientific_texts.txt",
        False,128,
        num_of_negative_samples=number_of_negative_samples
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_data.collate_fn,
        num_workers=4, pin_memory=True, drop_last=True
    )
    train_entities = train_data.entities
    training_parameters["train_data"] = train_data
    training_parameters["train_entities"] = train_entities
    training_parameters["train_loader"] = train_loader

    train_eval_loader = DataLoader(train_data, batch_size=EVAL_BATCH_SIZE, pin_memory=True)
    training_parameters["train_evaluation_loader"] = train_eval_loader

    if is_main_process():
        train_logger.info("Created training data dataset and loader")

    # --- Validation loader ---
    valid_data = TextWithGraphTopologyDataset(
        f"{path_to_triples}/validation.txt",
        graph_name, graph, TOKENIZER, "scientific_texts.txt", False,128
    )
    valid_loader = DataLoader(valid_data, batch_size=EVAL_BATCH_SIZE)
    training_parameters['validation_loader'] = valid_loader
    train_logger.info("Created validation data dataset and loader")

    # --- Test dataset ---
    test_data = TextWithGraphTopologyDataset(
        f"{path_to_triples}/test.txt",
        graph_name, graph, TOKENIZER, "scientific_texts.txt", False,128
    )

    # --- Entitiy sets ---
    validation_entities = valid_data.entities
    test_entities = test_data.entities
    train_validation_entities = list(set(train_entities.tolist()).union(set(validation_entities.tolist())))
    train_validation_test_entities = list(set(train_validation_entities).union(set(test_entities.tolist())))
    validation_new_entities = list(set(train_validation_entities).difference(set(train_entities.tolist())))
    test_new_entities = list(set(train_validation_test_entities).difference(set(train_validation_entities)))

    train_validation_entities = torch.tensor(train_validation_entities, dtype=torch.long)
    train_validation_test_entities = torch.tensor(train_validation_test_entities, dtype=torch.long)
    validation_new_entities = torch.tensor(validation_new_entities, dtype=torch.long)
    test_new_entities = torch.tensor(test_new_entities, dtype=torch.long)
    training_parameters['train_validation_entities'] = train_validation_entities
    mrr, model = train(training_parameters, hparameters_combo, train_logger, "augmented_foodon_scientific")


def evaluate_best_model(path_to_triples, graph, best_model_path):
    state = torch.load(best_model_path, map_location=DEVICE)
    train_logger = get_logger()
    train_logger.info("Loaded model and starting process")
    # Create the dataset objects and the dataloader objects
    train_data = data.TextWithGraphTopologyDataset(f"{path_to_triples}/train.txt",
                                                   "agrovoc",
                                                   graph,
                                                   TOKENIZER,
                                                   "definitions.txt",
                                                   False)

    valid_data = TextWithGraphTopologyDataset(f"{path_to_triples}/validation.txt",
                                              "agrovoc",
                                              graph,
                                              TOKENIZER,
                                              "definitions.txt",
                                              False,
                                              )
    valid_loader = DataLoader(valid_data, batch_size=EVAL_BATCH_SIZE)
    train_logger.info("Created validation data dataset and loader")
    test_data = TextWithGraphTopologyDataset(f"{path_to_triples}/test.txt",
                                             "agrovoc",
                                             graph,
                                             TOKENIZER,
                                             "definitions.txt",
                                             False,
                                             )
    test_loader = DataLoader(test_data, batch_size=EVAL_BATCH_SIZE)
    train_logger.info("Created test data dataset and loader")
    # Define hierarchical relation ids of the graph
    hierarchical_relations = list(
        set(train_data.hierarchical_relations).union(set(valid_data.hierarchical_relations)).union(
            set(test_data.hierarchical_relations)))

    # Code block for the creation of entity sets as part of assembling the graph
    # for filtered metrics computation

    train_entities = train_data.entities
    validation_entities = valid_data.entities
    test_entities = test_data.entities
    train_validation_entities = list(set(train_entities.tolist()).union(set(validation_entities.tolist())))
    train_validation_test_entities = list(set(train_validation_entities).union(set(test_entities.tolist())))
    test_new_entities_list = list(set(train_validation_test_entities).difference(set(train_validation_entities)))
    validation_new_entities_list = list(set(train_validation_entities).difference(set(train_entities.tolist())))
    train_validation_triples = torch.cat([train_data.triples, valid_data.triples])
    validation_graph_for_filtering = nx.DiGraph()
    validation_graph_for_filtering.add_weighted_edges_from(train_validation_triples.tolist())
    train_validation_triples = torch.cat([train_data.triples, valid_data.triples])
    validation_graph_for_filtering = nx.DiGraph()
    validation_graph_for_filtering.add_weighted_edges_from(train_validation_triples.tolist())

    train_validation_test_triples = torch.cat((train_data.triples, valid_data.triples, test_data.triples))
    test_graph_for_filtering = nx.DiGraph()
    test_graph_for_filtering.add_weighted_edges_from(train_validation_test_triples.tolist())

    # Make the lists tensors
    train_validation_entities = torch.tensor(train_validation_entities, dtype=torch.long)
    train_validation_test_entities = torch.tensor(train_validation_test_entities, dtype=torch.long)
    validation_new_entities = torch.tensor(validation_new_entities_list, dtype=torch.long)
    test_new_entities = torch.tensor(test_new_entities_list, dtype=torch.long)
    wl_role_index = train_data.wl_role_indices.max().item() + 1
    model = models.InductiveLinkPrediction(192,
                                           models.transe_score,
                                           models.margin_loss,
                                           5e-4,
                                           train_data.relation_description,
                                           0.15,
                                            wl_role_index,
                                           pred_graph_confidence=0.4).to(DEVICE)
    model.load_state_dict(state)

    train_logger.info("Filtered evaluation on validation set")
    number_of_val_inductive_triples = utils.compute_number_of_inductive_triples(valid_data.triples,
                                                                                validation_new_entities_list)
    filtered_validation_mrr = evaluate(valid_loader, model, train_data, train_validation_entities, train_logger,
                                       len(valid_loader), number_of_val_inductive_triples, hierarchical_relations,
                                       validation_graph_for_filtering, validation_new_entities)
    # train_logger.info("Filtered evaluation on validation dataset MRR: {:.4f}".format(filtered_validation_mrr))

    train_logger.info("Filtered evaluation on test set")

    number_of_test_inductive_triples = utils.compute_number_of_inductive_triples(test_data.triples,
                                                                                 test_new_entities_list)
    filtered_test_mrr = evaluate(test_loader, model, train_data, train_validation_test_entities, train_logger,
                                 len(test_loader), number_of_test_inductive_triples, hierarchical_relations,
                                 test_graph_for_filtering, test_new_entities)
    # train_logger.info("Filtered evaluation on test dataset MRR: {:.4f}".format(filtered_test_mrr))


def modality_dropout_scheduler(epoch, warmup=2, p_start=0.0, p_max=0.4, epochs_to_step=12):
    if epoch < warmup:
        return p_start
    step = min(epochs_to_step, max(0, epoch - warmup)) / max(1, epochs_to_step)

    return p_start + step * (p_max - p_start)


def predicted_confidence_scheduler(epoch, warmup=2, max_confidence=0.5, min_confidence=0.2, epochs_to_step=12):
    if epoch < warmup:
        return max_confidence
    step = min(epochs_to_step, epoch - warmup) / max(1, epochs_to_step)
    return max_confidence + step * (min_confidence - max_confidence)


def warmup_lambda(step: int, warmup_steps):
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank():
    return int(os.environ.get("RANK", "0"))


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    if not is_distributed():
        return
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(get_local_rank())


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def broadcast_float(x: float, device: torch.device) -> float:
    """Broadcast rank-0 float to all ranks."""
    if not is_distributed():
        return x
    t = torch.tensor([x], device=device, dtype=torch.float32)
    dist.broadcast(t, src=0)
    return float(t.item())

