from sklearn import datasets, model_selection, svm, metrics
import pandas as pd
# from sklearn.externals import joblib
import joblib
#import matplotlib.pyplot as plt
#import seaborn as sns
filename="/content/drive/MyDrive/iris.csv" #ファイル名
iris=pd.read_csv(filename) #データフレーム作成
iris_input=iris.iloc[:,0:4]
iris_label=pd.Series(data=iris["variety"])
input_train, input_test, label_train, label_test = model_selection.train_test_split(iris_input, iris_label)
clf = svm.SVC() #SVCのオブジェクトを生成
clf.fit(input_train, label_train) #学習
joblib.dump(clf, "svm.pkl", compress=True) # 学習済みモデルの保存
#filename = 'svm.sav'
#pickle.dump(model, open(filename, 'wb'))

predict_data = clf.predict(input_test) #学習済みのモデルに訓練データを入力し推論
#print(predict_data) #インプットデータに対して推定したラベルの出力
accuracy = metrics.accuracy_score(label_test, predict_data) #正解率の導出
print("Accuracy:",accuracy) #正解率
macro_f1 = metrics.f1_score(label_test, predict_data, average="macro") #マクロF1スコアの導出
micro_f1 = metrics.f1_score(label_test, predict_data, average="micro") #ミクロF1スコアの導出
print("Macro_F1:",macro_f1) #各F1スコア
print("Micro_F1:",micro_f1) #各F1スコア