from nltk.tokenize import word_tokenize


def flatten(t):
    "flattens a list of lists into a list"
    return [item for sublist in t for item in sublist]


def replace(l, old, occurence, new):
    for i in range(len(l)):
        if l[i] == old:
            if occurence == 0:
                l[i] = new
                break
            else:
                occurence -= 1
    return l




def str_replace(s, old, occurence, new):
    tokenized_list = word_tokenize(s)
    new_list = replace(tokenized_list, old, occurence, new)
    return " ".join(new_list)



def singular(word):
    if word[-1] == 's':
        return word[:-1]
    return word
    