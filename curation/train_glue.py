import argparse, pickle, re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_metric
from collections import namedtuple

from methods.likelihood import Likelihood
from methods.attention import Attention
from methods.representation import Representation



def tokenize(task_name):

    def tokenize_sst2(examples):
        return tokenizer(examples['sentence'], truncation=True, padding="max_length", max_length=args.max_len)

    def tokenize_cola(examples):
        return tokenizer(examples['sentence'], truncation=True, padding="max_length", max_length=args.max_len)

    def tokenize_mrpc(examples):
        return tokenizer(examples['sentence1'], examples["sentence2"], truncation=True, padding="max_length", max_length=args.max_len)

    def tokenize_stsb(examples):
        return tokenizer(examples['sentence1'], examples["sentence2"], truncation=True, padding="max_length", max_length=args.max_len)

    def tokenize_mnli(examples):
        return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding="max_length", max_length=args.max_len)

    def tokenize_rte(examples):
        return tokenizer(examples['sentence1'], examples["sentence2"], truncation=True, padding="max_length", max_length=args.max_len)

    def tokenize_wnli(examples):
        return tokenizer(examples['sentence1'], examples["sentence2"], truncation=True, padding="max_length", max_length=args.max_len)



    if task_name == "sst2":
        return tokenize_sst2
    if task_name == "cola":
        return tokenize_cola
    elif task_name == "stsb":
        return tokenize_stsb
    elif task_name == "mrpc":
        return tokenize_mrpc
    elif task_name == "mnli":
        return tokenize_mnli
    elif task_name == "rte":
        return tokenize_rte
    elif task_name == "wnli":
        return tokenize_wnli
    else:
        raise ValueError





def load_metadata(task_name):
    MetadataOutput = namedtuple("MetadataOutput", ['dataset_name', 'task_name', 'short_task_name', 'num_labels', 'remove_columns', 'cda_columns', 'validation'])
    if task_name == "sst2":
        return MetadataOutput(
            dataset_name="sst2",
            task_name="Sentiment Classification",
            short_task_name="sa",
            num_labels=2,
            remove_columns=["sentence", "idx"],
            cda_columns=["sentence"],
            validation="validation"
        )

    elif task_name == "cola":
        return MetadataOutput(
            dataset_name="cola",
            task_name="Linguistic Acceptability",
            short_task_name="cola",
            num_labels=2,
            remove_columns=["sentence", "idx"],
            cda_columns=["sentence"],
            validation="validation"
        )


    elif task_name == "mrpc":
        return MetadataOutput(
            dataset_name="mrpc",
            task_name="Paraphrase Detection",
            short_task_name="pd",
            num_labels=2,
            remove_columns=["sentence1", "sentence2", "idx"],
            cda_columns=["sentence1", "sentence2"],
            validation="validation"
        )

    elif task_name == "stsb":
        return MetadataOutput(
            dataset_name="stsb",
            task_name="Sementic Textual Similarity",
            short_task_name="sts",
            num_labels=1,
            remove_columns=["sentence1", "sentence2", "idx"],
            cda_columns=["sentence1", "sentence2"],
            validation="validation"
        )

    elif task_name == "mnli":
        return MetadataOutput(
            dataset_name="mnli",
            task_name="Senetence Entailment",
            short_task_name="se",
            num_labels=3,
            remove_columns=["premise", "hypothesis", "idx"],
            cda_columns=["premise", "hypothesis"],
            validation="validation_matched"
        )

    elif task_name == "rte":
        return MetadataOutput(
            dataset_name="rte",
            task_name="Sentence Entailment",
            short_task_name="rte",
            num_labels=2,
            remove_columns=["sentence1", "sentence2", "idx"],
            cda_columns=["sentence1", "sentence2"],
            validation="validation"
        )

    elif task_name == "wnli":
        return MetadataOutput(
            dataset_name="wnli",
            task_name="Sentence Entailment",
            short_task_name="wnli",
            num_labels=2,
            remove_columns=["sentence1", "sentence2", "idx"],
            cda_columns=["sentence1", "sentence2"],
            validation="validation"
        )
    
    else:
        raise ValueError




def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



def cda(examples, bias):
    if examples['idx'][0] not in most_biased_indices:
        # In this case, this example is not among the most biased. Return it as is. It's like keeping it in a filter
        return examples
    
    # Else, process this biased example. Either remove it, or do cda
    if not args.do_cda:
        # Filter out this example because we do not want to do CDA
        # To do that, set the idx at -1, then afterwards, filter the rows whose idx=-1
        examples["idx"] = [-1]
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
        examples["idx"] = [-1]
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
    parser.add_argument("--task", type=str, required=True, help="GLUE task to train", choices=["sst2", "cola", "stsb", "mrpc", "mnli", "wnli", "rte"])
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


    dataset_name, task_name, short_task_name, num_labels, remove_columns, cda_columns, validation = load_metadata(args.task)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = load_dataset('glue', args.task, cache_dir=args.cache_dir)
    
    print(dataset["train"])
    print()
    dataset["train"] = dataset["train"].map(lambda example: cda(example, args.bias_type), batched=True, batch_size=1, cache_file_name="{}/map/cda_{}_{}_{}_{}_{}".format(args.cache_dir, args.task, args.bias_type, args.remove_ratio, args.method, "cda" if args.do_cda else "nocda"))
    dataset["train"] = dataset["train"].filter(lambda example: example['idx'] != -1)
    print(dataset["train"])

    dataset = dataset.map(tokenize(args.task), batched=True, remove_columns=remove_columns, cache_file_names={k: "{}/map/tokenize_{}_{}_{}_{}_{}_{}".format(args.cache_dir, args.task, args.bias_type, args.remove_ratio, args.method, "cda" if args.do_cda else "nocda", k) for k in dataset.keys()})
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True, remove_columns=["label"], cache_file_names={k: "{}/map/rename_{}_{}_{}_{}_{}_{}".format(args.cache_dir, args.task, args.bias_type, args.remove_ratio, args.method, "cda" if args.do_cda else "nocda", k) for k in dataset.keys()})
    dataset.set_format(type='torch')

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
    metric = load_metric('glue', args.task)


    training_args = TrainingArguments(
        args.output,
        evaluation_strategy = "no" if args.task == "mnli" or not args.do_eval else "epoch",
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
        eval_dataset=dataset[validation],
        compute_metrics=compute_metrics
    )

    trainer.train()

    model.save_pretrained(args.output)

