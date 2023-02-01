import argparse, logging, pickle, statistics

def get_label_from_score(score):
    if score > args.threshold:
        return 1
    elif score < (-1 * args.threshold):
        return -1
    else: 
        return 0



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--likelihood", required=True, help="Filepath to bias scores of Likelihood method")
    parser.add_argument("--attention", required=True, help="Filepath to bias scores of Attention method")
    parser.add_argument("--representation", required=True, help="Filepath to bias scores of Representation method")
    parser.add_argument("--output", required=True, help="Output filepath")

    parser.add_argument("--method", default="combined", choices=["combined", "supervised"], help="Method to combined sub-scores of different bias computation methods")
    parser.add_argument("--threshold", default=0.03, help="Threshold above which to consider the input to be biased")

    args = parser.parse_args()

    print(vars(args))

    logger = logging.getLogger("transformers")
    logger.setLevel(level=logging.ERROR)

    # Read bias scores of Likelihood
    with open(args.likelihood, "rb") as f:
        likelihood = pickle.load(f)

    # Read bias scores of Attention
    with open(args.attention, "rb") as f:
        attention = pickle.load(f)

    # Read bias scores of Representation
    with open(args.representation, "rb") as f:
        representation = pickle.load(f)

    
    # Verify if all method beong to the same datasets
    assert likelihood["lm"] == attention["lm"]
    assert likelihood["lm"] == representation["lm"]

    assert likelihood["tokenizer"] == attention["tokenizer"]
    assert likelihood["tokenizer"] == representation["tokenizer"]

    assert len(likelihood["scores"].keys()) == len(attention["scores"].keys())
    assert len(likelihood["scores"].keys()) == len(representation["scores"].keys())

    # Create the new bias scores
    final_bias_scores = {
        "lm": likelihood["lm"],
        "tokenizer": likelihood["tokenizer"],
        "method": args.method,
        "threshold": args.threshold,
        "scores": {}
    }

    for idx in likelihood["scores"].keys():

        final_bias_scores["scores"][idx] = {}

        for bias_type in likelihood["scores"][idx].keys():

            if args.method == "combined":
                final_bias_scores["scores"][idx][bias_type] = statistics.mean([
                    likelihood["scores"][idx][bias_type],
                    attention["scores"][idx][bias_type],
                    representation["scores"][idx][bias_type]
                ])

            elif args.method == "supervised":
                mean_score = statistics.mean([
                    likelihood["scores"][idx][bias_type],
                    attention["scores"][idx][bias_type],
                    representation["scores"][idx][bias_type]
                ])

                final_bias_scores["scores"][idx][bias_type] = {
                    "likelihood": get_label_from_score(likelihood["scores"][idx][bias_type]),
                    "attention": get_label_from_score(attention["scores"][idx][bias_type]),
                    "representation": get_label_from_score(representation["scores"][idx][bias_type]),
                    "all": get_label_from_score(mean_score)
                }

            else:
                raise ValueError


    # Write the output
    with open(args.output, 'wb') as f:
        pickle.dump(final_bias_scores, f)