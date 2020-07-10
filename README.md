<!--
 * @Date: 2020-07-07 18:55:48
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2020-07-10 10:47:44
 * @FilePath: \kaggle\README.md
--> 
# MyKaggle competition
Titanic
- - - -
2020/07/10 

数据特征工程过程中增加特征值，同时在数据处理过程中对特征值进行筛选，基于筛选后的特征进行超参数优化，各算法预测平均精度
```python
The mean test score of model AdaBoostClassifier: 84.23118286084792
The mean test score of model BaggingClassifier: 87.65912465244382
The mean test score of model ExtraTreesClassifier: 85.83376965193737
The mean test score of model GradientBoostingClassifier: 86.63422804763955
The mean test score of model RandomForestClassifier: 87.01951883564644
The mean test score of model GaussianProcessClassifier: 86.25261583705843
The mean test score of model LogisticRegressionCV: 87.29241433657833
The mean test score of model BernoulliNB: 84.76411449943913
The mean test score of model GaussianNB: 82.94360711060608
The mean test score of model KNeighborsClassifier: 83.7351931770682
The mean test score of model SVC: 84.67602003316041
The mean test score of model XGBClassifier: 86.7671823882745
```
评分无法进一步提高，后续可针对特征工程进行更深层次的挖掘