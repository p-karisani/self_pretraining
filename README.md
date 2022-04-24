This is a Python code for the paper below: <br/>
*Semi-Supervised Text Classification via Self-Pretraining*, Payam Karisani, Negin Karisani. WSDM 2021. [Link](https://arxiv.org/abs/2109.15300)

**Pre-requirements**
- Python (>= 3.7.0)
- Numpy (>= 1.18.5)
- Pytorch (>= 1.4.0)

**Input**<br/>
The input files (train, test, or unlabeled) should contain one document per line. Each line should have 4 attributes (tab separated):
1) A unique document id (integer)
2) A binary label (integer):
	- The number 1 for negative documents
	- The number 3 for positive documents
	- If the document is unlabeled this column is ignored
3) Domain (string): a keyword describing the topic of the document
4) Document body (string)

See the file “sample.train” for a sample input.

**Training and Evaluation**<br/>
Below you can see an example command to run the code. This command tells the code to use a subset of the documents in the training and the unlabeled sets to train a model and evaluate in the test set—F1 measure is printed at the end of the execution.
```
python -m self_pretraining.src.MainThread --cmd bert_reg \
--itr 3 \
--model_path /user/desktop/bert-base-uncased/ \
--train_path /user/desktop/data/data.train \
--test_path /user/desktop/data/data.test \
--unlabeled_path /user/desktop/data/data.unlabeled \
--output_dir /user/desktop/output \
--device 0 \
--seed 666 \
--train_sample 500 \
--unlabeled_sample 10000 
```

The arguments are explained below:
- “--itr”: The number of iterations to run the experiment with different random seeds
- “--model_path”: The path to the huggingface pretrained bert
- “--train_path”: The path to the train file
- “--test_path”: The path to the test file
- “--unlabeled_path”: The path to the unlabeld file
- “--output_dir”: A directory to be used for temporary files
- “--device”: GPU identifier
- “--seed”: Random seed
- “--train_sample”: The number of documents to sample from the original labeled set to be used as the training data
- “--unlabeled_sample”: The number of unlabeled documents to sample from the unlabeled set to be used in the model

**Notes**
- The code uses the huggingface pretrained bert model: [Link](https://github.com/huggingface/transformers)
- The hyper-paremeters are set to their default values reported in the paper. You may need to change them, they are global values in the class “EPretrainProj”
- The batch size is set to 32.
