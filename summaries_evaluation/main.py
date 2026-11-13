from summary_information_evaluator import EvaluatorConfig, InformationalContentEvaluator
from summary_evaluator import TextTypeAwareSummaryEvaluator, EvaluationConfig
from utils import load_samples,load_texts
import json
from tqdm import tqdm

DATA_DIR = "Data/AGROVOC"
TEXT_TYPE = "scientific"

SAMPLES = load_samples(f"{DATA_DIR}/samples_{TEXT_TYPE}_texts.txt")

SOURCE_TEXTS = load_texts(f"{DATA_DIR}/long_descriptions.txt")

SUMMARIZED_TEXTS = load_texts(f"{DATA_DIR}/{TEXT_TYPE}_texts.txt")





def summary_evaluator(samples, source_texts,summarized_texts,txt_type):
    evaluations = {}
    config = EvaluationConfig.from_profile("balanced")
    config.text_type = txt_type

    evaluator = TextTypeAwareSummaryEvaluator(config)
    for sample in tqdm(samples):
        source = source_texts[sample]
        summary = summarized_texts[sample]
        report = evaluator.evaluate(source, summary)
        evaluations[sample] = report

    with open(f"{DATA_DIR}/revised_summmary_evaluations_{txt_type}.txt","w") as outfile:
        json.dump(evaluations, outfile)


if __name__ == "__main__":
   summary_evaluator(SAMPLES, SOURCE_TEXTS, SUMMARIZED_TEXTS, TEXT_TYPE)



