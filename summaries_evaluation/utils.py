import json

def load_samples(path_to_samples):
    samples = []
    with open(path_to_samples,"r") as file:
        for line in file.readlines():
            samples.append(line.strip())
    file.close()
    return samples


def load_texts(path_to_texts):
    return json.load(open(path_to_texts,"r"))