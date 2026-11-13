import statistics
import krippendorff as kr
import numpy as np
import pandas as pd
from prompt_toolkit.layout import dimension

DATA_DIR = "Data/FoodOn"

DIMENSIONS = [
    "Faithfulness",
    "Relevance",
    "Semantic consistency",
    "Informativeness"
]

JUDGES = {
    "OpenAI": f"{DATA_DIR}/judgements_openai.csv",
    "Anthropic": f"{DATA_DIR}/judgements_anthropic.csv",
    "Qwen": f"{DATA_DIR}/judgements_qwen.csv",
}



def krippendorff_alpha(judgements:pd.DataFrame):
    data = np.array(judgements.values.tolist()).T
    return kr.alpha(reliability_data=data,level_of_measurement="ordinal")

def load_judgements() -> dict[str, pd.DataFrame]:
    return {
        judge: pd.read_csv(path)
        for judge, path in JUDGES.items()
    }


def merge_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:

    merged = None

    for judge, df in data.items():

        copied = df[["URI"] + DIMENSIONS].copy()

        copied = copied.rename(columns={
            dimension: f"{dimension}_{judge}"
            for dimension in DIMENSIONS
        })

        if merged is None:
            merged = copied
        else:
            merged = merged.merge(
                copied,
                on="URI",
                how="inner",
                validate="one_to_one"
            )

    return merged


def compute_dimension_data(dimension: str, judgements_data: pd.DataFrame) -> pd.DataFrame:

    columns = [
        f"{dimension}_OpenAI",
        f"{dimension}_Anthropic",
        f"{dimension}_Qwen",
    ]

    dimension_data = pd.DataFrame(
        {
            "URI": judgements_data["URI"],
            "OpenAI": judgements_data[columns[0]],
            "Anthropic": judgements_data[columns[1]],
            "Qwen": judgements_data[columns[2]],
        }
    )

    assigned_scores = dimension_data[["OpenAI","Anthropic","Qwen"]]

    #Median value
    dimension_data['Median'] = assigned_scores.median(axis=1)

    #Disagreement analysis
    dimension_data['Scores range'] = assigned_scores.max(axis=1) - assigned_scores.min(axis=1)

    # Largest number of judges assigning exactly the same score
    dimension_data["Same count"] = assigned_scores.apply(
        lambda row: row.value_counts().max(),
        axis=1,
    )

    return dimension_data


def calculate_dimension_metrics(
    dimension: str,
    data: pd.DataFrame,
) -> dict:
    """Calculate agreement metrics for one dimension."""

    judgements = data[["OpenAI", "Anthropic", "Qwen"]]

    alpha = krippendorff_alpha(judgements)

    exact_agreement = (data["Same count"] == 3).mean()

    within_one_agreement = (data["Scores range"] <= 1).mean()

    mean_consensus = data["Median"].mean()

    return {
        "Dimension": dimension,
        "Krippendorff alpha": alpha,
        "Exact agreement": exact_agreement,
        "Within one agreement": within_one_agreement,
        "Mean consensus": mean_consensus,
    }



def main(save=False):

    #Load judgements
    judgement_data = load_judgements()

    #Table containing the judgements for every dimension from every judge, URI serves as indicator
    judgements = merge_data(judgement_data)

    dimensions_results = {}
    metrics = []

    #Get the data to extract analysis metrics for every dimension
    for dimension in DIMENSIONS:
        results = compute_dimension_data(dimension, judgements)
        dimensions_results[dimension] = results


        metrics.append(calculate_dimension_metrics(dimension, results))

    # Build consensus table
    consensus_data = pd.DataFrame({
        "URI": judgements["URI"],
        **{
            dimension: dimensions_results[dimension]["Median"]
            for dimension in DIMENSIONS
        },
    })


    # Aggregate metrics
    metrics_data = pd.DataFrame(metrics)

    if save:
       output_names = {
           "Faithfulness": "faithfulness_data.csv",
           "Relevance": "relevance_data.csv",
           "Semantic consistency": "semantic_consistency_data.csv",
           "Informativeness": "informativeness_data.csv",
       }

       for dimension, filename in output_names.items():
           dimensions_results[dimension].to_csv(
               f"{DATA_DIR}/{filename}",
               index=False,
           )

       consensus_data.to_csv(
           f"{DATA_DIR}/consensus_data.csv",
           index=False,
       )

       metrics_data.to_csv(
           f"{DATA_DIR}/metrics_data.csv",
           index=False,
       )


if __name__=="__main__":

    main(True)
