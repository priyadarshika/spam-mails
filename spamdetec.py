import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB , GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

data_frame = pd.read_csv ("spam.csv")

x = data_frame['EmailText']
y = data_frame['Label']

x_train , y_train = x[0:4457] , y[0:4457]
x_test, y_test = x[4457:],y[4457:]

cv = CountVectorizer()
features = cv.fit_transform(x_train)
tuned_parameter={'kernal':['linear','rbf'],'gama':[1e-3,1e-4],'c':[1,10,100,1000]}
model = GridSearchCV(svm.SVC(),tuned_parameter)

model = svm.SVC()

model.fit(features,y_train)

features_test = cv.transform (x_test)
if model.score(features_test,y_test) >0.8:
    print('mail is spam')
else:
    print('possibility of spam is: ',model.score(features_test,y_test))

