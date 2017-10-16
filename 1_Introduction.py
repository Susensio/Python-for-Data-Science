from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1
gnb = GaussianNB()
# 2
svc = SVC()
# 3
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 2), random_state=1)

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
gnb = gnb.fit(X, Y)
svc = svc.fit(X, Y)
mlp = mlp.fit(X, Y)

TEST = [[190, 70, 43]]

prediction = clf.predict(TEST)
prediction_gnb = gnb.predict(TEST)
prediction_svc = svc.predict(TEST)
prediction_mlp = mlp.predict(TEST)

# CHALLENGE compare their results and print the best one!

print("Tree: ", prediction)
print("score: ", clf.score(X, Y))
print("\nGNB: ", prediction_gnb)
print("score: ", gnb.score(X, Y))
print("\nSVC: ", prediction_svc)
print("score: ", svc.score(X, Y))
print("\nMLP: ", prediction_mlp)
print("score: ", mlp.score(X, Y))
