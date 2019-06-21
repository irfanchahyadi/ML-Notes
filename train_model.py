# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# TRAIN MODEL
model = LinearRegression()
model.fit(X_train, y_train)
