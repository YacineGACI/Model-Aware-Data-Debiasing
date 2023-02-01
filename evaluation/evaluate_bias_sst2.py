from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm
import math, argparse




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True, help="Full name or path or URL to tokenizer")
    parser.add_argument("--model", type=str, required=True, help="Full name or path or URL to trained NLI model")
    parser.add_argument("--eval_filepath", type=str, required=True, help="Filepath to evaluation templates")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_len", type=int, default=30, help="Maximum length allowed for input sentences. If longer, truncate.")

    args = parser.parse_args()
    print(vars(args))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the trained SST2 model
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model.to(device)

    softmax = torch.nn.Softmax(dim=1)


    sentiment = 0

    # Load the data
    with open(args.eval_filepath, 'r') as f:
        lines = f.readlines()[1:]
        num_iterations = math.ceil(len(lines) / args.batch_size)

        for batch in tqdm(range(num_iterations)):
            sentences_1 = []
            sentences_2 = []
            for line in lines[batch * args.batch_size: (batch + 1) * args.batch_size]:
                _, s1, s2 = line.rsplit(",", 2)
                sentences_1.append(s1.strip())
                sentences_2.append(s2.strip())

            inputs_1 = tokenizer(sentences_1, return_tensors="pt", padding="max_length", max_length=args.max_len)
            inputs_1 = {k:v.to(device) for k, v in inputs_1.items()}
            output_1 = model(**inputs_1)
            logits_1 = softmax(output_1.logits)

            inputs_2 = tokenizer(sentences_2, return_tensors="pt", padding="max_length", max_length=args.max_len)
            inputs_2 = {k:v.to(device) for k, v in inputs_2.items()}
            output_2 = model(**inputs_2)
            logits_2 = softmax(output_2.logits)

            for i in range(logits_1.size(0)):
                sentiment += abs(logits_1[i, 1].item() - logits_2[i, 1].item())

            
    print("Bias SST2: ", sentiment / len(lines))
    
