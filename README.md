# Repo for intent classification using search text queries. 

## Docker

To build a docker based service, clone the repo. The follow the below steps after changing into the directory

```docker build -t model-service .```

```docker run -p 8000:8000 model-service```

```curl -X GET -H "Content-Type: application/json" -d '{"texts":["Closest tv shop", "tv shops" , "i want to book a trip"] }' http://localhost:8000/predict```


## Train and test a new model

To train a new model 

```python src/train.py data/train_data.csv 0 1 --output_dir models/```

Test the new model

Try using the notebooks/RunModels.ipynb to see how to run the models at inference time.