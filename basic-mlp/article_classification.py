import numpy as np
import pandas as pd
from network_architecture.network import Network
from network_architecture.layers import FullyConnectedLayer, ActivationLayer
from activation_functions.sigmoid import sigmoid, sigmoid_prime
from error_functions.binary_cross_entropy import binary_cross_entropy, binary_cross_entropy_prime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV

dataset = pd.read_csv(r"datasets\article_level_data.csv")
dataset_final = dataset.drop("s.No", axis = 1)

print(dataset_final.head())
# print(dataset_final["class"].value_counts())

'''
0 - Human Generated
1 - AI generated
'''

vectorizer = CountVectorizer()
articles = vectorizer.fit_transform(dataset["article"]).toarray()

# After vectorizing the text, the dataset ended up with well over 10,000 features
# so we should reduce them using Principal Component Analysis to a certain number of features
# while retaining as much info as possible

# To find optimal number of components to reduce dataset to, we use GridSearchCV

'''
pca = PCA()
parameters = {'n_components': [50, 100, 500]}

searcher = GridSearchCV(estimator=pca, param_grid=parameters, cv=5).fit(x_train)
print(f"Best n_components: {searcher.best_params_['n_components']}")

Result was 500.
'''

pca = PCA(n_components=500)
articles = pca.fit_transform(articles)

x_train, x_test, y_train, y_test = train_test_split(articles, dataset["class"], test_size=0.3, random_state=42)

# print(x_train)

net =Network()

# Dealing with 500 features would also mean making a more sophisticated network architecture.

net.add_layer(FullyConnectedLayer(500, 64))
net.add_layer(ActivationLayer(sigmoid, sigmoid_prime))
net.add_layer(FullyConnectedLayer(64, 32))
net.add_layer(ActivationLayer(sigmoid, sigmoid_prime))
net.add_layer(FullyConnectedLayer(32, 16))
net.add_layer(ActivationLayer(sigmoid, sigmoid_prime))
net.add_layer(FullyConnectedLayer(16, 1))
net.add_layer(ActivationLayer(sigmoid, sigmoid_prime))

net.use_loss(binary_cross_entropy, binary_cross_entropy_prime)
net.fit(x_train, y_train.to_numpy().reshape(-1,1), epochs = 100, learning_rate=0.1)

# Predicting the output
out = net.predict(x_test)
# print(out)

# We want output as 0 or 1. So, we will be converting output array as such.
res = [arr.flatten().tolist() for arr in out]
res = [1 if arr[0] >=0.5 else 0 for arr in res]

test_values = y_test.to_numpy()
test_values = test_values.tolist()

'''
print(len(test_values))
print(len(res))
'''

# To find how many testing samples the neural network got right.
samples_right = 0
for i in range(len(test_values)):
    if test_values[i]==res[i]:
        samples_right+=1

print(f"Accuracy of the model on testing data is {samples_right*100/len(test_values):.3f}%\n(Model predicted {samples_right} test samples out of {len(test_values)} correctly.)")

# Getting user input, vectorizing, decomposing and returning classified output
def predict_user_input(user_input, vectorizer, pca, net):
    vec_test_string = vectorizer.transform([user_input]).toarray()

    vec_test_string_pca = pca.transform(vec_test_string)

    out = net.predict(vec_test_string_pca)
    
    res = 1 if out[0][0] >= 0.5 else 0
    return res

while True:
    user_input = input("Enter a string ('exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Bye bye!!")
        break

    result = predict_user_input(user_input, vectorizer, pca, net)
    if result == 0:
        print("The text is likely Human Generated")
    else:
        print("The text is likely AI Generated")
