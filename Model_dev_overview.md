
# Development approach

Training and tuning models for this large dataset was a but challenging due to the size. The model parameter tuning can be seen in notebooks folder. The final grid search for both models was done on a smaller sample. The model's label distribution is skewed, with the average at about 450. But there are several labels with much lower prevalence. 

## Whis is the type of model you selected for the problem and why.

I have tried with two models, 
1. first is a simple TF-IDF model to extract the textual representations from the text and then a LinearSVC classifier to classify. The model yields an average F-score of 0.62
2. The second model uses spacy embeddings topped with a MLP classifier to classify. This model performs <put numbers here >. 
The problems seems simple enought for this approach to work. The label distribution and the number of labels makes it a bit intractable to train. 

##  What are pre-processing steps that you took prior to training the model and why.
I tried to split the model into two models as low prevalence and high prevalence labels, but that created unnecessary complexity. Finally the current model is simple but effective on high prevalence labels. 

## What are methods you considered when evaluating the performance of model and why.

I evaluated in general based on F1 scores. There are simply too many labels types (1400) to drill down and evaluate individually in the limited time. 

## What are the weaknesses and possible improvements of the selected model for the given problem.

The key weakness here is the presence of large number of labels and modest samples per label. 
A better approach could be creating a heirarchical grouping of these labels and train an ensemble of models for each level of the heirarchy. This would provide a better control over individual groups of labels, and would allow us to change the model parameters based on the distribution of labels in each group. 

## What are architectural design that you would consider to manage and serve the model in the future.

The model architecture and size is simple at this point, so a simple docker image deployed in k8s to scale would be enough. But if this was done by heirarchical means, we would need cascades of models and an orchestrator to allow us to provide heirarchical inference. 