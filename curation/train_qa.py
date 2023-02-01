import argparse, pickle, re
from scipy.stats import mode
import numpy as np

from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from datasets import load_dataset, load_metric
from transformers import TrainingArguments, Trainer

from methods.likelihood import Likelihood
from methods.attention import Attention
from methods.representation import Representation


def tokenize(tokenizer):
    def inner(examples):
        return tokenizer(examples['question'], examples['context'], truncation=True, padding="max_length")

    return inner


def find_span_edges(context, answer):
    start_loop = context.index(102)
    j = 1
    new_span = True
    
    for i in range(start_loop, len(context) - 1):
        if context[i] == answer[j]:
            if new_span:
                start_pos = i
                new_span = False
            j += 1
        else:
            if j == len(answer) - 1:
                break
            else:
                j = 1
                new_span = True
    
    if j == len(answer) - 1: # If we found the span in the context
        end_pos = start_pos + len(answer) - 2
    else:
        start_pos = 0
        end_pos = 1

    return start_pos, end_pos



def add_start_end_positions(tokenizer):
    def inner(example):
        question = example["question"]
        context = example["context"]
        answers = example["answers"]["text"]
        start_positions = example["answers"]["answer_start"]

        # Find the most approved of value
        mode_value, _ = mode(start_positions)
        mode_index = start_positions.index(mode_value)
        answer = answers[mode_index]

        context_tokens = tokenizer(question, context)
        answer_tokens = tokenizer(answer)
        start_pos, end_pos = find_span_edges(context_tokens['input_ids'], answer_tokens['input_ids'])

        return {"start_positions": start_pos, "end_positions": end_pos}

    return inner





def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)





def cda(examples, bias):
    if examples['id'][0] not in most_biased_indices:
        # In this case, this example is not among the most biased. Return it as is. It's like keeping it in a filter
        return examples
    
    # Else, process this biased example. Either remove it, or do cda
    if not args.do_cda:
        # Filter out this example because we do not want to do CDA
        # To do that, set the idx at -1, then afterwards, filter the rows whose idx=-1
        examples["id"] = ["-1"]
        return examples
    
    # Here, we want to do CDA

    # Construct the patter for the regex
    # It's all the words that are present in the bias definition file
    pattern = "|".join(["\s" + k + "\s" for k, v in biasmeter.index.items() if v[0] == bias])

    # Find all words that match in all columns considered for cda.
    # e.g., for mnli: premise and hyothesis; for stsb: sentence1 and sentence 2; for sst2: sentence
    match = set(re.findall(pattern, examples[cda_columns[0]][0]))
    for col in cda_columns[1:]:
        match.update(set(re.findall(pattern, examples[col][0])))

    # If there is no match or more than one match, do nothing
    if len(match) != 1:
        # In traditional CDA, we do nothing to the example and we return it
        if args.remove_ratio == 1.0: # When this is the case, and --do_cda=True, this condition simulates traditional CDA
            return examples
        # However, with CDA to only biased examples, we can't let this example pass because it is biased
        # Then, we filter it out
        examples["id"] = ["-1"]
        return examples
    
    # If there is only one match, do CDA
    match = match.pop().strip() # Get the actual word, not the set
    candidates = [group for group in biasmeter.biases[bias].keys() if group != biasmeter.index[match][1]]

    word_replacement = re.compile(match, re.IGNORECASE)

    outputs = [[] for _ in range(len(cda_columns))]

    for i, col in enumerate(cda_columns):

        for s in examples[col]:        
            outputs[i] += [s] + [word_replacement.sub(c, s) for c in candidates]


    output_dict = {k: [v[0]] * (1 + len(candidates)) for k,v in examples.items()}

    for i, col in enumerate(cda_columns):
        output_dict[col] = outputs[i]

    return output_dict




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True, help="Full name or path or URL to tokenizer")
    parser.add_argument("--model", type=str, required=True, help="Full name or path or URL to trained NLI model")
    parser.add_argument("--task", type=str, required=True, help="SQUAD task to train", choices=["squad", "squad_v2"])
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum length allowed for input sentences. If longer, truncate.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight_decay")
    parser.add_argument("--output", type=str, required=True, help="Name of trained model (output). Will be saved in 'saved_models/tasks/$task/'")
    parser.add_argument("--do_eval", action="store_true", help="Do evaluation during training?")
    parser.add_argument("--no_cuda", action="store_true", help="No GPU?")
    parser.add_argument("--remove_ratio", type=float, default=0.0, help="Percetage of most stereotypes training instances to remove")
    parser.add_argument("--bias_scores", type=str, default=None, help="Pickled file to the bias scores per training example")
    parser.add_argument("--bias_type", type=str, default="gender", help="bias type to consider")
    parser.add_argument("--do_cda", action="store_true", help="True if wanting to use CDA, False if removal")
    parser.add_argument("--bias_def_filename", type=str, required=True, help="File to biases")

    parser.add_argument("--method", default="likelihood", choices=["likelihood", "attention", "representation"], help="Method to use for bias computation. Choice from: likelihood, attention or representation")

    parser.add_argument("--cache_dir", default="tmp/cache", help="Caching directory")

    args = parser.parse_args()
    print(vars(args))

    if args.method == "likelihood":
        biasmeter = Likelihood(lm=args.model, tokenizer=args.tokenizer, bias_def_filename=args.bias_def_filename, no_cuda=args.no_cuda)
    elif args.method == "attention":
        biasmeter = Attention(lm=args.model, tokenizer=args.tokenizer, bias_def_filename=args.bias_def_filename, no_cuda=args.no_cuda)
    else:
        biasmeter = Representation(lm=args.model, tokenizer=args.tokenizer, bias_def_filename=args.bias_def_filename, no_cuda=args.no_cuda)
        
    biasmeter.startup()

    # Read bias scores per idx
    with open(args.bias_scores, "rb") as f:
        scores = pickle.load(f)
    
    most_biased_indices = list(dict(sorted(scores["scores"].items(), key=lambda item: abs(item[1][args.bias_type]), reverse=True)).keys())
    most_biased_indices = most_biased_indices[:int(len(most_biased_indices) * args.remove_ratio)]

    cda_columns = ["context", "question"]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load the QA dataset
    dataset = load_dataset(args.task, cache_dir=args.cache_dir)

    print(dataset["train"])
    print()
    dataset["train"] = dataset["train"].map(lambda example: cda(example, args.bias_type), batched=True, batch_size=1, cache_file_name="{}/map/cda_{}_{}_{}_{}_{}".format(args.cache_dir, args.task, args.bias_type, args.remove_ratio, args.method, "cda" if args.do_cda else "nocda"))
    dataset["train"] = dataset["train"].filter(lambda example: example['id'] != "-1")
    print(dataset["train"])

    # Pre-process the dataset
    dataset = dataset.map(tokenize(tokenizer), batched=True, cache_file_names={k: "{}/map/tokenize_{}_{}_{}_{}_{}_{}".format(args.cache_dir, args.task, args.bias_type, args.remove_ratio, args.method, "cda" if args.do_cda else "nocda", k) for k in dataset.keys()})
    dataset = dataset.map(add_start_end_positions(tokenizer), cache_file_names={k: "{}/map/start_end_pos_{}_{}_{}_{}_{}_{}".format(args.cache_dir, args.task, args.bias_type, args.remove_ratio, args.method, "cda" if args.do_cda else "nocda", k) for k in dataset.keys()})
    columns = ['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions']
    dataset.set_format(type='torch', columns=columns)
    # dataset = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    print(dataset)

    # Load the model
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    metric = load_metric(args.task)


    training_args = TrainingArguments(
        args.output,
        evaluation_strategy = "no" if not args.do_eval else "epoch",
        learning_rate=args.lr,
        weight_decay=args.wd,
        no_cuda=args.no_cuda,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_strategy="no",
        num_train_epochs=args.epochs,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()

    model.save_pretrained(args.output)




