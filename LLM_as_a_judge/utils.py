import argparse

#Creates the triples list from txt files
def create_triples(path:str,encoding_str:str="utf-8"):
    """
    :param path: str
        The path of the file containing the graph in .txt format
    :param encoding_str: str
        The encoding of the file
    :return:
        A list of the triples contained in the graph
    """
    with open(path,encoding=encoding_str) as file:
        lines = []
        for line in file.readlines():
            lines.append(line.strip().split("\t"))

    return lines

#Splits the triple into its elements
def get_triple_elements(triples:list):
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

#Load the samples to evaluate
def load_samples(path_to_samples):
    samples = []
    with open(path_to_samples,"r") as file:
        for line in file.readlines():
            samples.append(line.strip())
    file.close()
    return samples


#Arguments parser for llm and graph selection
def get_args():
    parser = argparse.ArgumentParser(description="LLM as a judge process to evaluated AI generated texts")
    parser.add_argument("--llm", required=True, help="LLM that will be used to evaluate the generated texts.", choices = ["qwen","claude","chatgpt"])
    parser.add_argument("--graph", required=True, help="Graph file to be evaluated.", choices = ["agrovoc","foodon"])


    args = parser.parse_args()
    return args

