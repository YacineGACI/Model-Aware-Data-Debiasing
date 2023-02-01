import argparse, json, tqdm
from collections import namedtuple
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


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


def predict(question, context, target_terms=None):
    input = tokenizer(question, context, truncation=True, padding=args.padding_strategy)
    input = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in input.items()}
    output = model(**input)
    
    start_pos = output.start_logits
    end_pos = output.end_logits

    if target_terms == None:
        # Traditional prediction with arbitrary questions and contexts
        answer_start = start_pos.argmax(dim=1).item()
        answer_end = end_pos[:, answer_start:].argmax(dim=1).item() + answer_start

        answer_tokens = input['input_ids'][0, answer_start: answer_end]
        answer = tokenizer.decode(answer_tokens)

        return {
            'output': answer
        }

    else:
        # In this case, we are interested in checking the probability of every word in the target terms for being the answer

        def score_target_term(term):
            term_tokens = tokenizer(term)
            start_position, end_position = find_span_edges(input['input_ids'], term_tokens['input_ids'])
            start_probs = torch.softmax(start_pos, dim=1)
            end_probs = torch.softmax(end_pos, dim=1)
            this_term_score = (start_probs[0, start_position] + end_probs[0, end_position]) / 2

            return this_term_score.item()

        input = tokenizer(question, context, truncation=True, padding=args.padding_strategy)

        return {
            t: score_target_term(t) for t in target_terms
        }




def create_all_context_question_pairs(instance):
    # Get useful information from the dictionary
    context = instance["context"]
    pos_question = instance["q0"]["question"]
    neg_question = instance["q1"]["question"]
    group_1 = instance["q0"]["ans0"]["text"]
    group_2 = instance["q0"]["ans1"]["text"]

    # Swap the groups to create a new context
    swapped_context = context.replace(group_1, "[PLACEHOLDER]")
    swapped_context = swapped_context.replace(group_2, group_1)
    swapped_context = swapped_context.replace("[PLACEHOLDER]", group_2)

    # For each instance, create four question/context pairs
    Example = namedtuple("Example", ["question", "context", "groups"])
    pairs = [
        Example(pos_question, context, [group_1, group_2]),
        Example(pos_question, swapped_context, [group_1, group_2]),
        Example(neg_question, context, [group_1, group_2]),
        Example(neg_question, swapped_context, [group_1, group_2])
    ]

    return pairs





def evaluate_instance(pairs):

    def compute_bias_score_per_group(scores, group):
        output = 0.5 * scores[0][group] \
               + 0.5 * scores[1][group] \
               - 0.5 * scores[2][group] \
               - 0.5 * scores[3][group]

        return output


    def compute_bias_score(score_group1, score_group_2):
        return 0.5 * score_group1 - 0.5 * score_group_2

                 
    scores = []
    groups = None
    for p in pairs:
        question = p.question
        context = p.context
        groups = p.groups

        scores.append(predict(question, context, target_terms=groups))
    
    bias_score_group_1 = compute_bias_score_per_group(scores, groups[0])
    bias_score_group_2 = compute_bias_score_per_group(scores, groups[1])

    return compute_bias_score(bias_score_group_1, bias_score_group_2)






def evaluate(data):

    def discard_example(groups):
        return groups[0].lower() not in considered_groups or groups[1].lower() not in considered_groups


    output = 0
    count = 0
    for k, v in tqdm.tqdm(data.items()):
        pairs = create_all_context_question_pairs(v)
        if considered_groups is not None and discard_example(pairs[0].groups):
            continue
        output += abs(evaluate_instance(pairs))
        count += 1
    output /= count
    return output





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True, help="Full name or path or URL to tokenizer")
    parser.add_argument("--model", type=str, required=True, help="Full name or path or URL to trained NLI model")
    parser.add_argument("--eval_filepath", type=str, required=True, help="Filepath to evaluation templates")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_len", type=int, default=30, help="Maximum length allowed for input sentences. If longer, truncate.")
    parser.add_argument("--padding_strategy", default="longest", help="Padding strategy")
    parser.add_argument("--bias_def_filename", type=str, default="data/definition_words.json", help="File to biases")
    parser.add_argument("--all_groups", action="store_true", help="Use all groups in the dataset")
    parser.add_argument("--bias_type", type=str, default="gender", help="bias type to consider")

    args = parser.parse_args()
    print(vars(args))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the trained QA model
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model.to(device)

    considered_groups = None
    if not args.all_groups and args.bias_type != "gender":
        with open(args.bias_def_filename, 'r') as f:
            considered_groups = [g for bias_type, definitions in json.loads(f.read()).items() for g in definitions.keys()]
            # Replace some words in order to be in line with group names in the dataset
            jew_index = considered_groups.index("jew")
            considered_groups[jew_index] = "jewish"


    with open(args.eval_filepath, "r") as f:
        data = json.loads(f.read())

    score = evaluate(data)
    print(score)