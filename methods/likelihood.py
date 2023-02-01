import statistics, json

import torch
from transformers import pipeline
from nltk.tokenize import word_tokenize

import text_utils
from metrics import mean_substraction_metric as metric


class Likelihood:
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

        # Instantiate the language model
        self.nlp = pipeline(task="fill-mask", model=self.lm, tokenizer=self.tokenizer, device=self.device)




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




    def compute_masked_sentence_stereotype_simple(self, sentence, subgroup_word, bias_type, group):
        
        # Step 1: Get the groups constituing @bias_type without @group
        subgroups_words = [bias_type[k] for k in bias_type.keys() if k != group]
        subgroups_words = text_utils.flatten(subgroups_words) + [subgroup_word]
        
        # Step 2: Get probabilities for all words in subgroups_words
        output = self.nlp(sentence, targets=subgroups_words)

        # Step 4: Make these probabilities into a dictionary
        subgroups = {k:[] for k in bias_type.keys()}
        for o in output:
            try:
                token_group = self.index[o['token_str']][1]
                subgroups[token_group].append(o['score'])
            except:
                pass

        # Step 4bis: Take the max for every group
        for g in subgroups.keys():
            try:
                subgroups[g] = max(subgroups[g])
            except:
                subgroups[g] = 0


        # Step 5: Compute the total probability
        total_prob = sum([p for k,p in  subgroups.items()])
        
        # Step 6: Transform the probabilities of subgroups into a probability distribution
        for k in subgroups.keys():
            subgroups[k] = subgroups[k] / total_prob


        # Step 7: Compute the bias metric
        return metric(subgroups, self.index[subgroup_word][1])





    def compute_sentence_pll(self, sentence):
        words = word_tokenize(sentence)
        total_pll = 1
        for i, word in enumerate(words):
            masked_sentence_tokenized = [w if i != j else self.mask_token for j, w in enumerate(words)]
            masked_sentence = " ".join(masked_sentence_tokenized)
            
            output = self.nlp(masked_sentence, targets=[word])[0]
            total_pll *= output['score']

        return total_pll



    def compute_bias_score_per_masked_group_token(self, sentence, subgroup_word, bias_type, group):
        # Step 1: Get the groups constituing @bias_type without @group
        subgroups_words = [bias_type[k] for k in bias_type.keys() if k != group]
        subgroups_words = text_utils.flatten(subgroups_words) + [subgroup_word]
        
        # Step 2: Get probabilities for all words in subgroups_words
        output = {}
        for w in subgroups_words:
            input_sentence = sentence.replace("[GROUP]", w)
            output[w] = self.compute_sentence_pll(input_sentence)

        # Step 4: Make these probabilities into a dictionary
        subgroups = {k:[] for k in bias_type.keys()}
        for w in subgroups_words:
            try:
                token_group = self.index[w][1]
                subgroups[token_group].append(output[w])
            except:
                pass

        # Step 4bis: Take the max for every group
        for g in subgroups.keys():
            try:
                subgroups[g] = max(subgroups[g])
            except:
                subgroups[g] = 0


        # Step 5: Compute the total probability
        total_prob = sum([p for k,p in  subgroups.items()])
        
        # Step 6: Transform the probabilities of subgroups into a probability distribution
        for k in subgroups.keys():
            subgroups[k] = subgroups[k] / total_prob


        # Step 7: Compute the bias metric
        return metric(subgroups, self.index[subgroup_word][1])






    def run(self, sentence, likelihood_method="pll", *args, **kwargs):
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
        result = {k:[] for k in self.biases.keys()}


        # Step 3: Compute all biases in the queries
        for q_word, q_bias_type, q_group, q_occ in queries:
            # Replace q_word in the sentence with [MASK]
            masked_sentence = text_utils.str_replace(sentence, q_word, q_occ, self.mask_token if likelihood_method == "simple" else "[GROUP]")
            if likelihood_method == "simple":
                bias_score = self.compute_masked_sentence_stereotype_simple(masked_sentence, q_word, self.biases[q_bias_type], q_group)
            else:
                bias_score = self.compute_bias_score_per_masked_group_token(masked_sentence, q_word, self.biases[q_bias_type], q_group)
            result[q_bias_type].append(bias_score)


        # Step 4: For every demographic attribute, compute the mean of biases corresponding to every word
        for b in result.keys():
            if result[b] != []:
                result[b] = statistics.mean(result[b])
            else:
                result[b] = 0

        return result
    
