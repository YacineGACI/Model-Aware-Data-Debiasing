import statistics, json, itertools

import torch
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import word_tokenize

import text_utils
from metrics import mean_substraction_metric as metric



class Attention:
    def __init__(self, lm, tokenizer, bias_def_filename, no_cuda=False):
        self.lm = lm
        self.tokenizer = tokenizer
        self.bias_def_filename = bias_def_filename
        self.device = 0 if not no_cuda and torch.cuda.is_available() else -1
        self.mask_token = "<mask>" if "roberta" in self.lm else "[MASK]"


    def startup(self):
        print("cuda:{}".format(self.device) if self.device >= 0 else "cpu")

        # Read the social groups lexicons
        with open(self.bias_def_filename, 'r') as f:
            self.biases = json.load(f)

        # Create the index for definitional words
        self.index = self.create_inverted_index()

        # Load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.model = AutoModel.from_pretrained(self.lm)
        self.model.to(self.device)




    def read_bias_file(self):
        '''
            Reads @filename and returns a dictionary of biases. This dictionary is organized as follows:
            gender   =>   male         =>   male, he, him, his, himself, brother, father...
                        female       =>   female, she, her, herself, sister, mother...
            race     =>   white        =>   caucasian, white, europe, european, america, american
                    =>   black        =>   black, africa, african, afro, ghana
                    =>   asian        =>   asian, asia, oriental, japan, china, korea, japanese

        '''
        biases = {}
        with open(self.bias_def_filename, 'r') as f:
            for line in f.readlines():
                bias_dimension, social_group, lexicon = line.split('\t')
                bias_dimension = bias_dimension.strip()
                social_group = social_group.strip()
                lexicon = [w.strip() for w in lexicon.split(',')]

                if bias_dimension in biases.keys():
                    biases[bias_dimension][social_group] = lexicon
                else:
                    biases[bias_dimension] = {social_group: lexicon}
        return biases


    def create_inverted_index(self):
        '''
            Creates an inverted index for definitial words.
            Each word maps to a tuple (bias_type, group). For example:
                man    ==> (gender, male)
                muslim ==> (religion, islam)
        '''
        index = {}
        for bias_type in self.biases.keys():
            for group, def_words in self.biases[bias_type].items():
                for w in def_words:
                    index[w] = (bias_type, group)
        return index





    def find_group_position(self, input_ids, groups):
        # Find the [SEP] position. All group positions must be after the sep id
        sep_pos = input_ids.index(self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token))

        group_ids = self.tokenizer.convert_tokens_to_ids(groups)
        return {groups[i]: input_ids.index(group_ids[i], sep_pos) for i in range(len(groups))}
        



    def aggregate_attention(self, attentions, group_positions, num_heads=12, layer=-1, target_pos=0):
        scores = {k: 0 for k in group_positions.keys()}

        # Take the corresponding layer of attentions
        attentions = attentions[layer]

        for head in range(num_heads):
            for group, group_pos in group_positions.items():
                scores[group] += attentions[0, head, target_pos, group_pos].item()

        scores = {k: v/num_heads for k, v in scores.items()}
        return scores






    def compute_attention_bias(self, masked_sentence, groups, group, do_permutations=True, target="cls", num_heads=12, layer=-1):
        # Step 0: Initialize the result
        result = {g: [] for g in groups}
        
        # Step 1: Compute all permutations for groups
        group_permutations = list(itertools.permutations(groups))
        if not do_permutations:
            group_permutations = group_permutations[0:1]

        for groups in group_permutations:
            # Step 2: Create an artificial sentence made of groups
            second_input = " ".join(groups)

            # Step 3: Tokenize the masked sentence as Input 1 and an artificial list of groups as Input 2
            input = self.tokenizer(masked_sentence, second_input)

            # Step 4: Find the positions of group-related words in the input
            group_positions = self.find_group_position(input["input_ids"], groups)
            
            # Step 5: Find the position of the target vector [CLS] or [MASK]
            target_pos = input["input_ids"].index(self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)) if target == "mask" else 0

            # Step 6: Transform the iput to tensors
            for k, v in input.items():
                input[k] = torch.tensor(v).unsqueeze(0).to(self.device)

            # Step 7: Get the attentions
            attentions = self.model(**input, output_attentions=True).attentions

            # Step 8: Take the mean of attention heads of the last layer
            scores = self.aggregate_attention(attentions, group_positions, num_heads, layer, target_pos)

            for g in groups:
                result[g].append(scores[g])
            
        # Step 9: Take the mean of bias scores across al permutations
        for k, v in result.items():
            result[k] = statistics.mean(v)

        return metric(result, group)

        




    def run(self, sentence, do_permutations=True, target="cls", num_heads=12, layer=-1, *args, **kwargs):
        # Step 1: Find words in @sentence that are among the bias seed words and create corresponding queries
        #         Queries are of the form (word, bias_type, group, occurence_in_sentence)
        #         For example: (man, gender, male, 0), (muslim, religion, muslim, 0), (white, race, european, 1)...
        #         ocucurence_in_sentence are used when the same word is in the sentence more than once

        queries = []
        sentence = sentence.lower()
        definitional_words = {} # This is to keep track of how many times a definitional word is in @sentence
        for w in word_tokenize(sentence):
            w_new = text_utils.singular(w)
            if w_new in self.index.keys():
                if w_new not in definitional_words.keys():
                    definitional_words[w_new] = 0
                else:
                    definitional_words[w_new] += 1
                queries.append((w_new, self.index[w_new][0], self.index[w_new][1], definitional_words[w_new]))
                sentence = text_utils.str_replace(sentence, w, 0, w_new)
        

        # Step 2: Instantiate the result dict
        result = {k: [] for k in self.biases.keys()}

        # Step 3: Compute all biases in the queries
        for q_word, q_bias_type, q_group, q_occ in queries:
            # Replace q_word in the sentence with [MASK]
            masked_sentence = text_utils.str_replace(sentence, q_word, q_occ, self.mask_token)
            groups = list(self.biases[q_bias_type].keys())
            bias_score = self.compute_attention_bias(masked_sentence, groups, q_group, do_permutations, target, num_heads, layer)
            result[q_bias_type].append(bias_score)


        # Step 4: For every demographic attribute, compute the mean of biases corresponding to every word
        for b in result.keys():
            if result[b] != []:
                result[b] = statistics.mean(result[b])
            else:
                result[b] = 0

        return result
    
