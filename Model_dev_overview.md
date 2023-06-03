# Intent classification from text. 
The two notebooks show my approach in solving the intent classification problem from text. 

## The approach :
I decided to take two approaches to solve this problem 

- [1] One fairly simple approach, which uses Tf-Idf vectorization to convert text into vectors, and then trains logistic regression on top

- [2] A bit more complicated which uses spacy pretrained model to convert text into embeddings and then train an MLP classifier on top. 

Both approaches deliver results, but for the the TFIDF approach wins by a large margin, and that too considering the simpler lighter structure of the model. 

# Model Details. 

Lets dive into the individual models. Both models are built using similar recipies: 
- The first part is the model design with model description. 
- Then the model optimisation using Grid search using 5 fold cross validation.
- Then a complete fit using the best parameters. 
- Then some performance analysis using average precision, confusion, and other per intent metrics. 
- Finally a complete fit on the training data with saved model output and model binary. 


## TF-IDF model 
This is a simple model that converts text into vectors using TF-IDF. It then fit logistic regression classifiers for each of the intent using One Vs Rest scheme. 

- The model attains a 5 fold CV accuracy of 93.5 % 
- The model generates an average weighted precision of 94%, recall of 94%, and F1-score of 94% 
- The model's worst performing class is restaurant reservation, most confused with cancel_reservation and confirm_reservation


## Spacy model 
This model uses spacy pretrained model to extract embeddings from the text, and then trains a Multi Layered Perceptron on top. 

- The model attains a 5 fold CV accuracy of 78%
- The model generates an average weighted precision of 80%, recall of 79%, and F1-score of 79%  

- the model's worst performing intent is "no", most commonly confused with yes, maybe, and who_made_you



# Model outputs and binaries. 
- The model inference outputs are saved in the outputs directory 
- The model binaries are saved in the models directory