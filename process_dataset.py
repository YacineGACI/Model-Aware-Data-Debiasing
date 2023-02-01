import argparse, logging, pickle, tqdm
from statistics import mean
from datasets import load_dataset
from methods.likelihood import Likelihood
from methods.attention import Attention
from methods.representation import Representation


def sentence_bias_score(input_sentences):
    all_dicts = []
    for s in input_sentences:
        all_dicts.append(biasmeter.run(s, likelihood_method=args.l_method, do_permutations=args.do_permutations, target=args.target, num_heads=args.num_heads, layer=args.layer))
    
    possible_keys = all_dicts[0].keys()
    result = {}

    for k in possible_keys:
        result[k] = sum([d[k] for d in all_dicts]) / len(all_dicts)

    return result
        




def dataset_metedata(dataset):
    if dataset == "sst2":
        return "idx", ["sentence"]
    
    if dataset == "mnli":
        return "idx", ["premise", "hypothesis"]
    
    if dataset == "stsb":
        return "idx", ["sentence1", "sentence2"]

    if dataset == "squad":
        return "id", ["context", "question"]

    if dataset == "squad_v2":
        return "id", ["context", "question"]






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", default="bert-base-uncased", help="Language model")
    parser.add_argument("--tokenizer", default="bert-base-uncased", help="Tokenizer")
    parser.add_argument("--bias_def_filename", default="data/definition_words.json", help="Filename to definition words")
    parser.add_argument("--dataset", default="sst2", help="GLUE dataset", choices=["sst2", "stsb", "mnli", "squad", "squad_v2"])
    parser.add_argument("--output", required=True, help="Filepath to output")
    parser.add_argument("--no_cuda", action="store_true", help="No GPU?")

    parser.add_argument("--method", default="likelihood", choices=["likelihood", "attention", "representation"], help="Method to use for bias computation. Choice from: likelihood, attention or representation")

    # Specific to likelihood-based bias computation
    parser.add_argument("--l_method", default="simple", help="Likelihood calculation method", choices=["pll", "simple"])

    # Specific to attention-based bias computation
    parser.add_argument("--do_permutations", action="store_true", help="Compute attention-based bias on al possible permutations of social groups?")
    parser.add_argument("--target", default="cls", help="Which attention vector to consider?", choices=["cls", "mask"])
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads of consider")
    parser.add_argument("--layer", type=int, default=-1, help="Which layer of the model to consider?")


    args = parser.parse_args()

    print(vars(args))

    logger = logging.getLogger("transformers")
    logger.setLevel(level=logging.ERROR)

    if args.method == "likelihood":
        biasmeter = Likelihood(lm=args.lm, tokenizer=args.tokenizer, bias_def_filename=args.bias_def_filename, no_cuda=args.no_cuda)
    elif args.method == "attention":
        biasmeter = Attention(lm=args.lm, tokenizer=args.tokenizer, bias_def_filename=args.bias_def_filename, no_cuda=args.no_cuda)
    else:
        biasmeter = Representation(lm=args.lm, tokenizer=args.tokenizer, bias_def_filename=args.bias_def_filename, no_cuda=args.no_cuda)
    biasmeter.startup()

    bias_scores = {
        "lm": args.lm,
        "tokenizer": args.tokenizer,
        "method": args.l_method,
        "scores": {}
    }

    if args.dataset in ["sst2", "cola", "mnli", "wnli", "rte", "mrpc", "stsb"]:
        dataset = load_dataset('glue', args.dataset, split="train")
    else:
        dataset = load_dataset(args.dataset, split="train")
    for d in tqdm.tqdm(dataset):
        try:
            id_label, input_labels = dataset_metedata(args.dataset) # Names of input sentences in the dataset
            input = [d[i] for i in input_labels] # Actual inut sentences
            idx = d[id_label]
            bias_scores["scores"][idx] = sentence_bias_score(input)
        except:
            bias_scores["scores"][idx] = {k: 0 for k in biasmeter.biases.keys()}

    with open(args.output, 'wb') as f:
        pickle.dump(bias_scores, f)

    
