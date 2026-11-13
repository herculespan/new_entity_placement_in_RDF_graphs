import json
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
from anthropic import Anthropic
from tqdm import tqdm
import ollama
from utils import *


load_dotenv("variables.env")
API_KEY = os.getenv("CHAT_GPT_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SYSTEM_PROMPT = """You are an evaluator of automatically generated natural-language definitions for entities in a knowledge graph.

## TASK

A target entity did not have an available textual definition. A language model generated a definition for that entity using information extracted from the entity's local knowledge-graph context.

Your task is to assess the quality of the GENERATED DEFINITION strictly with respect to the supplied GRAPH CONTEXT.

Evaluate the generated definition independently along four dimensions:

1. Faithfulness
2. Relevance
3. Semantic consistency
4. Informativeness

Assign an integer score from 1 to 5 for each dimension according to the rubrics below.

IMPORTANT EVALUATION PRINCIPLES:

* Treat the supplied graph context as the primary and authoritative evidence for this evaluation.
* Do NOT evaluate the definition primarily according to your external or pretrained world knowledge.
* Do NOT introduce external facts in order to justify a higher or lower score.
* A statement that may be true in the real world but is not supported or conservatively inferable from the supplied graph context must not be treated as graph-supported evidence.
* Evaluate relation direction carefully. A relation from the target entity to another entity is not necessarily semantically equivalent to the inverse relation.
* Distinguish the target entity from its neighboring entities. Information describing a neighbor is not automatically information describing the target.
* Conservative semantic inference from clearly supplied hierarchical or relational evidence is permitted, but speculative inference is not.
* Evaluate each dimension independently. A high or low score on one dimension must NOT automatically determine the scores of the other dimensions.
* Do not assign scores according to an overall impression of fluency or writing quality.
* Do not reward verbosity.
* Do not penalize a concise definition merely because it is short if it captures the salient information available in the graph.
* When graph context is sparse, informativeness must be evaluated relative to the useful information actually available, not relative to an ideal encyclopedic definition.
* Do not penalize the definition for omitting graph facts that are redundant, peripheral, or not useful for defining the target entity.
* Do not produce an overall quality score. The four dimensions must remain separate.

The generated definition may therefore legitimately receive very different scores across the four dimensions.

For example, a definition may be completely faithful, relevant, and semantically consistent but insufficiently informative. Such a case could appropriately receive scores such as:

Faithfulness = 5
Relevance = 5
Semantic consistency = 5
Informativeness = 2

## DIMENSION 1 — FAITHFULNESS

### Evaluation question

To what extent are the substantive claims made in the generated definition supported by, conservatively inferable from, and non-contradictory with the supplied graph context?

The purpose of this dimension is to detect unsupported claims, hallucinated attributes, incorrect relational interpretations, and contradictions with the graph evidence.

### Scale

**5 — Fully faithful**
All substantive claims in the definition are supported by or conservatively inferable from the supplied graph context. No substantive unsupported claims or contradictions are present.

**4 — Highly faithful**
The definition is almost completely supported by the graph context. It may contain a minor peripheral inference or slightly stronger wording than the evidence warrants, but its central characterization is fully supported and no important contradiction is present.

**3 — Moderately faithful**
The definition is generally supported by the graph context, but one or more claims are questionable, weakly supported, or extend beyond what can reasonably be inferred. The central characterization remains broadly compatible with the evidence.

**2 — Mostly unfaithful**
Several important claims are unsupported or inconsistent with the graph context, or an important aspect of the definition is based on an unjustified inference. Only part of the definition is well grounded.

**1 — Completely or predominantly unfaithful**
The central characterization of the target entity is unsupported, contradicted by the graph context, or largely fabricated from information not available in the supplied evidence.

When assigning this score, ask:

"Is what the definition says actually supported by the supplied graph evidence?"

---

## DIMENSION 2 — RELEVANCE

### Evaluation question

To what extent does the generated definition specifically and directly characterize the TARGET ENTITY rather than discussing neighboring, overly general, peripheral, or unrelated concepts?

### Scale

**5 — Fully relevant**
Every substantive part of the definition directly contributes to characterizing the target entity. The definition is focused and contains no meaningful tangential or irrelevant material.

**4 — Highly relevant**
Nearly all of the definition directly characterizes the target entity. Any peripheral or overly general information is minor and does not substantially reduce focus.

**3 — Moderately relevant**
The definition clearly concerns the target entity, but noticeable portions are generic, tangential, overly broad, or focused on contextual concepts rather than on the target itself.

**2 — Mostly irrelevant**
The definition has only a weak or partial focus on the target entity. Much of its content describes neighboring entities, broad contextual concepts, or peripheral information rather than defining the target.

**1 — Completely or predominantly irrelevant**
The definition fails to meaningfully characterize the target entity. It primarily describes another entity or concept, or is so generic or unrelated that it provides essentially no target-specific characterization.

When assigning this score, ask:

"How much of this definition is actually about the target entity?"

---

## DIMENSION 3 — SEMANTIC CONSISTENCY

### Evaluation question

To what extent is the overall meaning of the generated definition semantically compatible with the target entity's category, hierarchical position, relation directions, and relationships represented in the supplied graph context?

This dimension concerns conceptual compatibility, not merely whether individual statements are explicitly present in the graph.

### Scale

**5 — Fully semantically consistent**
The definition is completely compatible with the target entity's semantic category, hierarchical position, and supplied relationships. No meaningful conceptual conflict or semantic ambiguity is identifiable.

**4 — Highly semantically consistent**
The overall semantic interpretation is compatible with nearly all graph evidence. Any conceptual ambiguity or inconsistency is minor and does not alter the essential semantic characterization of the entity.

**3 — Moderately semantically consistent**
The definition is broadly compatible with the graph context, but some wording creates semantic ambiguity or minor incompatibilities concerning category, hierarchy, relation interpretation, or conceptual role.

**2 — Substantially inconsistent**
The definition contains one or more major semantic incompatibilities with the graph context. It may assign the entity an inappropriate semantic category, role, hierarchical interpretation, or relational meaning, although some aspects remain compatible.

**1 — Fundamentally semantically inconsistent**
The definition's central interpretation conflicts with the semantic representation of the entity in the graph. For example, it may characterize a process as an organism, a disease as a pathogen, a substance as a location, or otherwise fundamentally misidentify what kind of entity the target represents.

When assigning this score, ask:

"Does this definition conceptually make sense given what the graph indicates this entity is and how it relates to other entities?"

---

## DIMENSION 4 — INFORMATIVENESS

### Evaluation question

To what extent does the generated definition make effective use of the salient, useful, and discriminative information available in the graph context to characterize the target entity?

Evaluate informativeness RELATIVE TO THE INFORMATION AVAILABLE IN THE SUPPLIED GRAPH CONTEXT.

A longer definition is not automatically more informative.

The definition is not required to reproduce every graph relation. Redundant, peripheral, or non-definitional graph information may appropriately be omitted.

### Scale

**5 — Fully informative**
The definition captures all or nearly all salient, useful, and non-redundant information available in the supplied context that is needed to meaningfully characterize and distinguish the target entity.

**4 — Highly informative**
The definition captures most of the salient contextual information needed to understand the entity. Only secondary, redundant, or relatively minor useful information is omitted.

**3 — Moderately informative**
The definition captures some important characteristics of the entity and provides a reasonable basic characterization, but it omits a substantial amount of useful or discriminative information available in the graph context.

**2 — Slightly informative**
The definition communicates only limited useful information about the entity. It may identify a broad category or one basic property but fails to exploit most of the salient evidence available in the graph context.

**1 — Uninformative**
The definition provides essentially no useful characterization of the entity despite the supplied graph evidence. It is vacuous, tautological, excessively generic, or fails to convey meaningful available information.

When assigning this score, ask:

"How effectively does the definition use the useful information actually available in the graph context?"

---

## IMPORTANT DISTINCTIONS BETWEEN DIMENSIONS

Keep the following distinctions explicit throughout the evaluation:

FAITHFULNESS:
"Is what it says supported?"

RELEVANCE:
"Is what it says about the right entity?"

SEMANTIC CONSISTENCY:
"Does its conceptual interpretation fit the entity's graph position and relationships?"

INFORMATIVENESS:
"Does it capture enough of the useful available information?"

A definition can be relevant but unfaithful.

A definition can be faithful but insufficiently informative.

A definition can contain individually plausible statements but still assign the entity the wrong semantic type and therefore be semantically inconsistent.

Do not force the four scores to be similar.

---

## EVALUATION PROCEDURE

Perform the evaluation in the following order:

1. Examine the target entity and graph context.
2. Examine the generated definition.
3. Evaluate FAITHFULNESS using only the faithfulness rubric.
4. Independently evaluate RELEVANCE using only the relevance rubric.
5. Independently evaluate SEMANTIC CONSISTENCY using only the semantic-consistency rubric.


When moving from one dimension to another, do not treat the previous score as evidence for the next score.

Provide concise evidence-based rationales. Do not provide an extended chain of thought. The rationale should state the principal observable reason for the score and, where relevant, identify the supplied graph evidence that supports the judgment.

---

## OUTPUT FORMAT

Return ONLY one valid JSON object with exactly the following structure:

{
"faithfulness": {
"score": 1,
"rationale": "Concise evidence-based explanation."
},
"relevance": {
"score": 1,
"rationale": "Concise evidence-based explanation."
},
"semantic_consistency": {
"score": 1,
"rationale": "Concise evidence-based explanation."
},
"informativeness": {
"score": 1,
"rationale": "Concise evidence-based explanation."
}
}

CONSTRAINTS:

* Every score must be an integer from 1 through 5.
* Do not return fractional scores.
* Do not calculate an overall or average score.
* Do not add fields not specified in the schema.
* Do not wrap the JSON in Markdown code fences.
* Do not mention the identity of the language model performing the evaluation.
* Base all substantive judgments on the supplied graph evidence.
"""




class PromptPreparation:
    def __init__(self,path_to_triples,path_to_labels,path_to_definitions,graph_name):
        self.triples = create_triples(path_to_triples)
        self.heads, self.relations, self.tails = get_triple_elements(self.triples)
        self.heads_array = np.array(self.heads)
        self.tails_array = np.array(self.tails)
        self.labels = json.load(open(path_to_labels,"r"))
        self.definitions = json.load(open(path_to_definitions,"r"))
        self.hierarchical = "broader" if graph_name.lower() == "agrovoc" else "subClassOf"
    def create_prompt(self,sample):
        label = self.labels[sample]
        definition = self.definitions[sample]
        narrower_concepts = set()
        broader_concepts = set()
        related_concepts = set()
        head_indices = np.where(self.heads_array == sample)[0]
        tail_indices = np.where(self.tails_array == sample)[0]
        for head_index in head_indices:
            head = self.heads[head_index]
            relation = self.relations[head_index]

            tail = self.tails[head_index]
            if self.hierarchical in relation:
                broader_concepts.add(tail)
            else:
                related_concepts.add(tail)

        for tail_index in tail_indices:
            tail = self.tails[tail_index]
            head = self.heads[tail_index]
            relation = self.relations[tail_index]
            if self.hierarchical in relation:
                narrower_concepts.add(head)
            else:
                related_concepts.add(head)

        broader_text = "\n".join(f"- {concept}" for concept in broader_concepts)
        narrower_text = "\n".join(f"- {concept}" for concept in narrower_concepts)
        related_text = "\n".join(f"- {concept}" for concept in related_concepts)

        user_prompt = f"""
                                Entity ID:
                                {sample}
                                Entity label:
                                {label}
                                GRAPH CONTEXT
                                The graph context below is the evidence available for evaluating the generated definition.
                                Relations are directional and are represented explicitly.
                                Broader concepts:
                                {broader_text}

                                Narrower concepts:
                                {narrower_text}

                                Other relations:
                                {related_text}

                                GENERATED DEFINITION
                                {definition}
                                Evaluate this generated definition according to the previously specified four-dimensional rubric and return only the required JSON object.
                """

        return user_prompt


class LLMJudge:
    def __init__(self,llm_client,provider=None):
        self.client = llm_client
        self.provider = provider

    def judge(self,user_prompt,model_id):
        if self.provider.lower() == "openai":
            response = self.client.responses.create(
                model = model_id,
                instructions = SYSTEM_PROMPT,
                input = user_prompt,
                tools = [],
                tool_choice = "none",
                store=False
            )
            return response.output_text
        elif self.provider.lower() == "anthropic":
            response = self.client.messages.create(
                model = model_id,
                max_tokens = 1500,
                system = SYSTEM_PROMPT,
                messages = [
                    {
                        "role":"user",
                        "content": user_prompt,
                    }
                ],

            )
            judge_output = "".join(block.text for block in response.content if block.type =="text")
            return judge_output
        elif self.provider.lower() == "qwen":
            response = self.client.chat(
                model="qwen3:32b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={"temperature": 0.0},
                stream=False,
            )
            judge_output = response["message"]["content"]
            return judge_output
        return "No provider given"




def main():
    arguments = get_args()
    llm = arguments.llm
    graph = arguments.graph

    match llm:
        case "chatgpt":
            model = "gpt-5.6-sol"
            client = OpenAI(api_key=API_KEY)
            provider = "openai"
        case "qwen":
            model="qwen3:32b"
            client = ollama.Client(host="http://localhost:11434")
            provider = "qwen"
        case "claude":
            model = "claude-opus-5"
            client = Anthropic(api_key=CLAUDE_API_KEY)
            provider = "anthropic"

        case _:
            raise ValueError("Unsupported LLM was selected as a judge")

    match graph:
        case "agrovoc":
            triples_path = "Data/AGROVOC/agrovoc_2020_concepts.txt"
            labels_path = "Data/AGROVOC/entities2labels.txt"
            definitions_path = "Data/AGROVOC/definitions.txt"
            samples_path = "Data/AGROVOC/agrovoc_sample_200.txt"
            output_path = f"Data/AGROVOC/agrovoc_sample_{provider}_judgements.txt"
        case "foodon":
            triples_path = "Data/FoodOn/foodon_classes_2711_clean.txt",
            labels_path = "Data/FoodOn/entities2labels.txt",
            definitions_path = "Data/FoodOn/definitions.txt",
            samples_path = "Data/FoodOn/foodon_sample_200_entities_no_blank_links.txt"
            output_path = f"Data/FoodOn/foodon_sample_{provider}_judgements.txt"

        case _:
            raise ValueError("Unsupported graph name passed")


    preparator = PromptPreparation(
        path_to_triples=triples_path,
        path_to_labels=labels_path,
        path_to_definitions= definitions_path,
        graph_name=graph)
    llm_judge = LLMJudge(client, provider)
    samples = load_samples(samples_path)
    judgements = {}
    for sample in tqdm(samples):
        prompt = preparator.create_prompt(sample)
        res = llm_judge.judge(prompt, model)
        judgements[sample] = res

        print(judgements)
        print("=" * 80)

    with open(output_path, "w") as outfile:
        json.dump(judgements, outfile)



if __name__ == "__main__":

    main()








