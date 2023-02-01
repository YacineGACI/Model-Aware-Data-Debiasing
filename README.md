# Towards Debiasing Task-Specific NLP Models by Debiasing their Training Data
This implementation is the source code companion of our paper entitled *"Towards Debiasing Task-Specific NLP Models by Debiasing their Training Data"*. This project contains code to identify which instances in data risk amplifying social stereotypes in the finetuning phase, and also code to actually finetune downstream NLP models with either removing or augmenting the most problematic instances in training data.

## Requirements
- datasets==1.1.2
- nltk==3.5
- numpy==1.18.4
- scikit_learn==1.1.1
- scipy==1.4.1
- torch==1.8.1
- tqdm==4.46.0
- transformers==4.12.5



## Computing bias scores for a dataset
The core of the paper is about identifying which instances in training data concur most with the stereotypes encoded in text encoders. To do so, we provide code to compute bias scores for each data instance. Interpretations of bias scores are as follows:
- **== 0**: there is no social stereotypes detected in the data instance. The latter is assumed safe to use in training.
- **> 0**: the data instance mentions a stereotype that is already encoded in the text encoder. Using this data instance in training might amplify this prejudice in the final finetuned model.
- **< 0**: the data instance mentions a stereotype. However, the text encoder encodes the semantic opposite of that stereotype (anti-stereotype). Using this data instance in training helps re-balancing the encoded prejudice in models. However, too many of such instances might backfire by forcing anti-stereotypes to be learned by the final models, which is also bad.

Also, the intensity of bias scores gives an idea about how strongly the associated stereotypes are encoded. For example, a data instance with a bias score of 0.9 is much more dangerous to use than an instance with a score of 0.3. 

To compute bias scores for every data instance, run the following command:

`python process_dataset.py --lm <lm> --tokenizer <tokenizer> --bias_def_filename <filepath> --dataset <dataset> --output <output> --method <method>`

- In our experiments, we use *bert-base-uncased* as a language model and a tokenizer.
- <filepath> points to the JSON file describing bias types, social groups and their definition words. The default value is *data/definition_words.json*
- for datasets, we use the datasets provided by [HuggingFace Data Hub](https://huggingface.co/datasets). In the current version, only these datasets are supported: *sst2*, *stsb*, *mnli*, *squad*, *squad_v2*
- the output should specify the filename where bias scores for each data instance will be saved.
- We support three different methods to excavate bias information from text encoders: *likelihood*, *attention*, and *representation*.



### Combining bias scores of all three methods
In our experiments, we test our debiasing pipeline with bias scores resulting from the combination of all three methods (i.e. likelihoods, attention weights and vector representations). To do that, run the following:

`python combine_labeling_methods.py --likelihood <path_likelihood_scores> --attention <path_attention_scores> --representation <path_representation_scores> --output <path_output>`



## Finetuning NLP models on task-specific datasets
After having computed bias scores for every instance in the training data, it is time to finetune your favorite NLP model on this dataset. If the task is a GLUE task, run the following command:

`python -m curation.train_glue --tokenizer <tokenizer> --model <model> --task <task> --output <output> --remove_ratio <ratio> --bias_scores <bias_scores_filepath> --bias_type <bias_type> --do_cda --bias_def <bias_def_filename> --method <method>`

- **model**: the name of the text encoder to finetune
- **tokenizer**: name of tokenizer to use
- **task**: name of glue task. The following are supported: sst2, cola, stsb, mrpc, mnli, wnli, rte
- **output**: filepath to the output where the final model will be saved
- **ratio**: the ratio of instances in the training data to remove or augment.
- **bias_scores_filepath**: path to the pickled file containing bias scores for every instance in the data. This is usually the output of the first command described in this file.
- **bias_type**: either gender, race or religion
- **bias_def_filename**: path to file containig definition of biases and social groups
- **method**: either likelihood, attention or representation
- **--do_cda**: if set, do augment the most biased training instances instead of removing them.

To train a question answering model, run the following: 
`python -m curation.train_qa --tokenizer <tokenizer> --model <model> --task <task> --output <output> --remove_ratio <ratio> --bias_scores <bias_scores_filepath> --bias_type <bias_type> --do_cda --bias_def <bias_def_filename> --method <method>`

The parameters are the same, except for **task**, which takes the vlaues of either *squad*, or *squad_v2*.



## Evaluation
To evaluate the fairness aspects of finetuned models, we provide a separate source file for every task we include in our experiments and that are included in the paper.

### Sentence Inference
`python evaluation/evaluate_bias_nli.py --tokenizer <tokenizer> --model <model> --eval_filepath <eval_filepath>`

### Sentiment Classification
`python evaluation/evaluate_bias_sst2.py --tokenizer <tokenizer> --model <model> --eval_filepath <eval_filepath>`

### Question Answering
`python evaluation/evaluate_bias_qa.py --tokenizer <tokenizer> --model <model> --eval_filepath <eval_filepath> --bias_def_filename <bias_def_filename> --bias_type <bias_type>`

you can find the evaluation datasets in the Github repositories published by their respective authors.
- sentence inference and sentiment classification: https://github.com/sunipa/On-Measuring-and-Mitigating-Biased-Inferences-of-Word-Embeddings
- question answering: https://github.com/allenai/unqover