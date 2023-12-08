# opensource_final_project
It contains sklearn machine learning model classifying tumor datasets of opensoureSW Final Project.

## training dataset
```
// make MRI scan to 64X64 digit array by contrast colors
image_size = 64
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']

images = []
y = []
for i in labels:
    folderPath = os.path.join('./tumor_dataset/Training',i)
    for j in os.listdir(folderPath):
        img = skimage.io.imread(os.path.join(folderPath,j),)
        img = skimage.transform.resize(img,(image_size,image_size))
        img = skimage.color.rgb2gray(img)
        images.append(img)
        y.append(i)
        
images = np.array(images)

X = images.reshape((-1, image_size**2))
y = np.array(y)
```

## modeling
```
// construct basic models by using KNN and SVC with handeling parameters.
knn = KNeighborsClassifier(n_neighbors=1)
svc = SVC(C=10, kernel='rbf', gamma=1.5, probability=True)

// bagging each models to improve
bagged_knn = BaggingClassifier(base_estimator=knn, n_estimators=10, random_state=42, n_jobs=-1)
bagged_svc = BaggingClassifier(base_estimator=svc, n_estimators=10, random_state=42, n_jobs=-1)

// by voting each bagging model create main model with handeling parameters.
model = VotingClassifier(estimators=[('bagged_knn', bagged_knn), ('bagged_svc', bagged_svc)], voting='soft', weights=[1, 150], n_jobs=-1)

// fitting main model
model.fit(X_train, y_train)

// predict model
y_pred = np.zeros_like(y_test)
y_pred = model.predict(X_test)

print('Accuracy: %.2f' % sklearn.metrics.accuracy_score(y_test, y_pred))
```

## techniques to reason appropriate parameters
```
// used cofusin matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'])

// visualization
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

## note
***YOU NEED TO RUN CODE WITH PICKLE FILE***

## copyright and license
Copyright (c) 2023 중앙대22 이성욱
MIT License

## contact info
E-mail: bill8342@naver.com
