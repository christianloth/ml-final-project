# -*- coding: utf-8 -*-

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import statistics
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_curve, classification_report, roc_curve, auc
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as pyplot


custom_rf_parameters = [
    {'numberOfEstimators': 125, 'maxDepthOfTree': 5, 'minNodeSizeToSplit': 18, 'maxFeaturesToBeSelected': 3},
    {'numberOfEstimators': 80, 'maxDepthOfTree': 6, 'minNodeSizeToSplit': 4, 'maxFeaturesToBeSelected': 12},
    {'numberOfEstimators': 152, 'maxDepthOfTree': 13, 'minNodeSizeToSplit': 17, 'maxFeaturesToBeSelected': 18}
]

ada_parameters = [
    {'n_estimators': 50, 'learning_rate': 0.5},
    {'n_estimators': 100, 'learning_rate': 1.0},
]

xgb_parameters = [
    {'n_estimators': 100, 'learning_rate': 0.1},
    {'n_estimators': 200, 'learning_rate': 0.05},
]

bag_parameters = [
    {'n_estimators': 5},
    {'n_estimators': 20},
]

"""Data source: https://www.kaggle.com/datasets/yakhyojon/national-basketball-association-nba/data"""

dataFrame = pd.read_csv("https://raw.githubusercontent.com/Nava308/basketball/main/nba-players.csv")
dataFrame.head(10)

features = ['blk', 'tov','gp', 'min','fta', 'ft', 'oreb', '3p_made', '3pa', '3p',
       'ftm','reb', 'ast', 'stl','dreb', 'fga', 'fg',   'pts', 'fgm']

class Utilities:
  @staticmethod
  def countNullValuesByColumn(dataFrame):
    return dataFrame.isnull().sum()

  @staticmethod
  def countDuplicateRows(dataFrame):
    return dataFrame.duplicated().sum()

  @staticmethod
  def countNullValuesOverall(dataFrame):
    return dataFrame.isnull().sum().sum()

  @staticmethod
  def describeDataFrame(dataFrame):
    print("##########################################################################")
    print(dataFrame.dtypes)
    print("##########################################################################")
    print(dataFrame.info())
    print("##########################################################################")
    print(dataFrame.describe())
    print("##########################################################################")

  @staticmethod
  def removeDuplicates(dataFrame):
    return dataFrame.drop_duplicates()

  @staticmethod
  def getCorrelationMatrix(dataFrame):
    return dataFrame.corr(numeric_only=True).round(2)

  @staticmethod
  def plot_heatmap_for_correlation_matrix(dataFrame):
    plot.figure(figsize=(10, 8))
    correlationMatrix = Utilities.getCorrelationMatrix(dataFrame)
    sns.heatmap(correlationMatrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plot.xticks(range(dataFrame.select_dtypes(['number']).shape[1]), dataFrame.select_dtypes(['number']).columns, fontsize=7, rotation=42)
    plot.yticks(range(dataFrame.select_dtypes(['number']).shape[1]), dataFrame.select_dtypes(['number']).columns, fontsize=7)
    plot.title('Correlation Matrix', fontsize=11)
    plot.show()

  @staticmethod
  def getHighlyCorrelatedColumns(dataFrame):
    epsilon = 0.7
    correlationMatrix = Utilities.getCorrelationMatrix(dataFrame)
    col_count = len(correlationMatrix.columns)
    columns = np.full(col_count, True, dtype=bool)

    for i in range(col_count):
      for j in range(i+1, col_count):
        if correlationMatrix.iloc[i,j] > epsilon:
          columns[j] = False
    return columns

  @staticmethod
  def normalizeColumn(col):
    maxVal = max(col)
    minVal = min(col)
    avgVal = np.mean(col)
    normalizedVals = [(x - avgVal)/(maxVal-minVal) for x in col]
    return normalizedVals

  @staticmethod
  def normalize(dataFrame):
    return dataFrame.apply(Utilities.normalizeColumn,axis=0)

  @staticmethod
  def testTrainSplit(dataFrame,predictionLabel):
    trainingRows = int(np.floor(len(dataFrame)*0.9))
    dataFrame = dataFrame.sample(frac=1, random_state=3)
    XTrain = dataFrame[features][:trainingRows]
    yTrain = dataFrame[predictionLabel][:trainingRows].values
    XTest = dataFrame[features][trainingRows:]
    yTest = dataFrame[predictionLabel][trainingRows:].values
    return XTrain,yTrain,XTest,yTest

Utilities.countNullValuesByColumn(dataFrame)

Utilities.countNullValuesOverall(dataFrame)

Utilities.countDuplicateRows(dataFrame)

Utilities.describeDataFrame(dataFrame)

print(len(dataFrame))

print(len(Utilities.removeDuplicates(dataFrame)))

df_with_duplicates = dataFrame
dataFrame = Utilities.removeDuplicates(dataFrame)

Utilities.getCorrelationMatrix(dataFrame)

Utilities.plot_heatmap_for_correlation_matrix(dataFrame)

Utilities.getHighlyCorrelatedColumns(dataFrame)

Utilities.normalize(dataFrame.select_dtypes(include='number'))

dataFrame.columns

"""Entropy and Information gain"""

class DecisionTree:
  @staticmethod
  def entropy(probability,hasAnySpecialConditions=False):
    if probability == 0 or probability == 1:
          return 0
    else:
        return - (probability * np.log2(probability) + (1 - probability) * np.log2(1-probability))

  @staticmethod
  def informationGain(left,right,hasAnySpecialConditions=False):
    parent = left + right
    pp = 0
    if len(parent) > 0:
      pp = parent.count(1) / len(parent)
    pl = 0
    if len(left) > 0:
      pl = left.count(1) / len(left)
    pr= 0
    if len(right) > 0:
      pr = right.count(1) / len(right)
    informationGainParent = DecisionTree.entropy(pp)
    informationGainLeft = DecisionTree.entropy(pl)
    informationGainRight = DecisionTree.entropy(pr)
    return informationGainParent - len(left) / len(parent) * informationGainLeft - len(right) / len(parent) * informationGainRight

  @staticmethod
  def _findBestSplit(X, y, maxFeaturesToBeSelected):
    featursList = list()
    totalFeatures = len(X[0])

    while len(featursList) <= maxFeaturesToBeSelected:
      featureIndex = random.sample(range(totalFeatures), 1)
      if featureIndex not in featursList:
          featursList.extend(featureIndex)

      bestInformationGain = -9999
      node = None
      for featureIndex in featursList:
        for splitPoint in X[:,featureIndex]:
            leftChild = {'X': [], 'y': []}
            rightChild = {'X': [], 'y': []}

            if type(splitPoint) == int or  type(splitPoint)==float:
                for idx, val in enumerate(X[:,featureIndex]):
                    if val <= splitPoint:
                        leftChild['X'] += [X[idx]]
                        leftChild['y'] += [y[idx]]
                    else:
                        rightChild['X'] += [X[idx]]
                        rightChild['y'] += [y[idx]]

            else:
                for idx, val in enumerate(X[:,featureIndex]):
                    if val == splitPoint:
                        leftChild['X'] += [X[idx]]
                        leftChild['y'] += [y[idx]]
                    else:
                        rightChild['X'] += [X[idx]]
                        rightChild['y'] += [y[idx]]

            splitInformationGain = DecisionTree.informationGain(leftChild['y'], rightChild['y'])
            if splitInformationGain > bestInformationGain:
                bestInformationGain = splitInformationGain
                leftChild['X'] = np.array(leftChild['X'])
                rightChild['X'] = np.array(rightChild['X'])
                node = {'informationGain': splitInformationGain,
                        'leftChild': leftChild,
                        'rightChild': rightChild,
                        'splitPoint': splitPoint,
                        'featureIndex': featureIndex}
    return node

  @staticmethod
  def findBestSplit(X, y, maxFeaturesToBeSelected):
    bestInformationGain = -9999
    node = None
    n_samples, n_features = X.shape

    # Select subset of features randomly
    features = np.random.choice(n_features, maxFeaturesToBeSelected, replace=True)

    for featureIndex in features:
      # Sort the data along selected feature
      sorted_indices = np.argsort(X[:, featureIndex])
      sorted_X, sorted_y = X[sorted_indices], y[sorted_indices]

      for i in range(1, n_samples):
        # only consider split points where the value changes
        if sorted_y[i] != sorted_y[i - 1]:
          splitPoint = (sorted_X[i, featureIndex] + sorted_X[i - 1, featureIndex]) / 2  # get middle X point
          left_y, right_y = sorted_y[:i], sorted_y[i:]

          splitInformationGain = DecisionTree.informationGain(list(left_y), list(right_y))

          if splitInformationGain > bestInformationGain:
            bestInformationGain = splitInformationGain
            left_indices, right_indices = sorted_indices[:i], sorted_indices[i:]
            node = {'informationGain': splitInformationGain,
                    'leftChild': {'X': X[left_indices], 'y': y[left_indices]},
                    'rightChild': {'X': X[right_indices], 'y': y[right_indices]},
                    'splitPoint': splitPoint,
                    'featureIndex': featureIndex}

    return node

  @staticmethod
  def terminalTreeNode(child):
    y = child['y']
    prediction = statistics.mode(y)
    return prediction

  @staticmethod
  def _splitTreeNode(node, maxFeatures, minSampleSplit, maxDepth, depth,hasAnySpecialConditions=False):
    leftChild = node['leftChild']
    rightChild = node['rightChild']

    del(node['leftChild'])
    del(node['rightChild'])

    if len(leftChild['y']) == 0 or len(rightChild['y']) == 0:
        emptyChild = {'y': leftChild['y'] + rightChild['y']}
        node['leftSplit'] = DecisionTree.terminalTreeNode(emptyChild)
        node['rightSplit'] = DecisionTree.terminalTreeNode(emptyChild)
        return

    if depth >= maxDepth:
        node['leftSplit'] = DecisionTree.terminalTreeNode(leftChild)
        node['rightSplit'] = DecisionTree.terminalTreeNode(rightChild)
        return node

    if len(leftChild['X']) <= minSampleSplit:
        node['leftSplit'] = node['rightSplit'] = DecisionTree.terminalTreeNode(leftChild)
    else:
        node['leftSplit'] = DecisionTree.findBestSplit(leftChild['X'], leftChild['y'], maxFeatures)
        DecisionTree.splitTreeNode(node['leftSplit'], maxDepth, minSampleSplit, maxDepth, depth + 1)
    if len(rightChild['X']) <= minSampleSplit:
        node['rightSplit'] = node['leftSplit'] = DecisionTree.terminalTreeNode(rightChild)
    else:
        node['rightSplit'] = DecisionTree.findBestSplit(rightChild['X'], rightChild['y'], maxFeatures)
        DecisionTree.splitTreeNode(node['rightSplit'], maxFeatures, minSampleSplit, maxDepth, depth + 1)

  @staticmethod
  def splitTreeNode(node, maxFeatures, minSampleSplit, maxDepth, depth,hasAnySpecialConditions=False):
    if node is None:
      return

    # Extract left and right children from the current node. Return None if key doesn't exist
    leftChild = node.get('leftChild')
    rightChild = node.get('rightChild')

    # If either child is None, no further splitting is possible
    if leftChild is None or rightChild is None:
      return

    # Base case: if the tree has reached its maximum depth
    if depth >= maxDepth:
      node['leftSplit'] = DecisionTree.terminalTreeNode(leftChild)
      node['rightSplit'] = DecisionTree.terminalTreeNode(rightChild)
      return

    # Handle the left child
    if len(leftChild['y']) <= minSampleSplit:
      node['leftSplit'] = DecisionTree.terminalTreeNode(leftChild)
    else:
      node['leftSplit'] = DecisionTree.findBestSplit(leftChild['X'], leftChild['y'], maxFeatures)
      DecisionTree.splitTreeNode(node['leftSplit'], maxFeatures, minSampleSplit, maxDepth, depth + 1)

    # Handle the right child
    if len(rightChild['y']) <= minSampleSplit:
      node['rightSplit'] = DecisionTree.terminalTreeNode(rightChild)
    else:
      node['rightSplit'] = DecisionTree.findBestSplit(rightChild['X'], rightChild['y'], maxFeatures)
      DecisionTree.splitTreeNode(node['rightSplit'], maxFeatures, minSampleSplit, maxDepth, depth + 1)


  @staticmethod
  def buildTree(X, y, maxDepth, minSamplesSplit, maxFeatures):
    rootNode = DecisionTree.findBestSplit(X, y, maxFeatures)
    DecisionTree.splitTreeNode(rootNode, maxFeatures, minSamplesSplit, maxDepth, 1)
    return rootNode

  @staticmethod
  def predictTree(tree, X):
    featureIndex = tree['featureIndex']
    # print(type(X[featureIndex]), X[featureIndex])
    # print(type(tree['splitPoint']), tree['splitPoint'])
    if X[featureIndex] <= tree['splitPoint']:
        if type(tree['leftSplit']) != dict:
            return tree['leftSplit']
        else:
            return DecisionTree.predictTree(tree['leftSplit'], X)
    else:
        if type(tree['rightSplit']) != dict:
           return tree['rightSplit']
        else:
           return DecisionTree.predictTree(tree['rightSplit'], X)

class BootStrap:
  @staticmethod
  def bootstrapSample(X, y):
    selectedIndices = list(np.random.choice(range(len(X)), len(X), replace = True))
    outOfBagIndices = [i for i in range(len(X)) if i not in selectedIndices]
    XTrain = X.iloc[selectedIndices].values
    yTrain = y[selectedIndices]
    XTest = X.iloc[outOfBagIndices].values
    yTest = y[outOfBagIndices]
    return XTrain, yTrain, XTest, yTest

  @staticmethod
  def testError(model,X,y):
    predictions = [DecisionTree.predictTree(model, X) for X in X]
    missClassifiedPoints = 0
    for p,l in zip(predictions,y):
      if p!=l:
        missClassifiedPoints+=1
    print("Error ", missClassifiedPoints/len(X))
    return missClassifiedPoints/len(X)

class RandomForest:
  @staticmethod
  def buildRandomForestTree(X,y,number_of_estimators,max_depth_of_tree,min_node_size_to_split,max_features_to_be_selected):
    decision_tree_list = list()
    error_list = list()

    for cur_estimator in range(number_of_estimators):
      XTrain,yTrain,XTest,yTest = BootStrap.bootstrapSample(X,y)
      cur_tree = DecisionTree.buildTree(XTrain,yTrain,max_depth_of_tree,min_node_size_to_split,max_features_to_be_selected)
      decision_tree_list.append(cur_tree)
      error_list.append(BootStrap.testError(cur_tree,XTest,yTest))

    print("Test error (Out of bag estimate): {:.2f}".format(np.mean(error_list)))
    return decision_tree_list

  @staticmethod
  def predictRandomForest(treeList, XTest,hasAnySpecialConditions=False):
    predList = list()
    for i in range(len(XTest)):
        ensemblePreds = [DecisionTree.predictTree(tree, XTest.values[i]) for tree in treeList]
        finalPred = max(ensemblePreds, key = ensemblePreds.count)
        predList.append(finalPred)
    return np.array(predList)

"""Here is the graph and printing of our custom random forest implementation without sklearn"""

def print_table(dataFrame):
    colWidths = {}

    for columns in dataFrame.columns:
        maxLength = max(dataFrame[columns].astype(str).apply(len).max(), len(columns))
        colWidths[columns] = maxLength

    headerRow = ""
    for columns in dataFrame.columns:
        headerRow += columns.ljust(colWidths[columns]) + " | "
    headerRow = headerRow[:-3]

    dividerRow = ""
    for columns in dataFrame.columns:
        dividerRow += "-" * colWidths[columns] + "-+-"
    dividerRow = dividerRow[:-3]

    print(headerRow)
    print(dividerRow)

    for i, r in dataFrame.iterrows():
        rowString = ""
        for columns in dataFrame.columns:
            rowString += str(r[columns]).ljust(colWidths[columns]) + " | "
        rowString = rowString[:-3]
        print(rowString)

def run_and_plot_custom_random_forest(custom_rf_parameters):
  res = []
  rocsForPlotting = []

  for idx, params in enumerate(custom_rf_parameters):
    numberOfEstimators = params['numberOfEstimators']
    maxDepthOfTree = params['maxDepthOfTree']
    minNodeSizeToSplit = params['minNodeSizeToSplit']
    maxFeaturesToBeSelected = params['maxFeaturesToBeSelected']

    XTrain, yTrain, XTest, yTest = Utilities.testTrainSplit(dataFrame, "target_5yrs")
    model = RandomForest.buildRandomForestTree(XTrain, yTrain, numberOfEstimators, maxDepthOfTree, minNodeSizeToSplit, maxFeaturesToBeSelected)
    predictions = RandomForest.predictRandomForest(model, XTest)

    # Replace None with a random choice of 0 or 1, if any Nones exist
    def replace_none_with_random_binary(array):
      return [random.choice([0, 1]) if x is None else x for x in array]

    yTest= replace_none_with_random_binary(yTest)
    predictions = replace_none_with_random_binary(predictions)

    # Calculate metrics
    acc = accuracy_score(yTest, predictions)
    pr = precision_score(yTest, predictions)
    re = recall_score(yTest, predictions)
    fStat = f1_score(yTest, predictions)

    res.append({
      "Experiment Number": idx + 1,
      "Parameters (Model)": f"Estimators: {numberOfEstimators}, Depth: {maxDepthOfTree}, Min Split: {minNodeSizeToSplit}, Max Features: {maxFeaturesToBeSelected}",
      "Results (Accuracy, Precision, Recall, F1-Score)": f"Accuracy: {acc:.3f}, Precision: {pr:.3f}, Recall: {re:.3f}, F1-Score: {fStat:.3f}"
    })

    # Compute ROC metrics and store
    falsePositiveRate, truePositiveRate, thresholds = roc_curve(yTest, predictions)
    areaUnderCurve = auc(falsePositiveRate, truePositiveRate)
    rocsForPlotting.append((falsePositiveRate, truePositiveRate, areaUnderCurve))

  # Plot all ROC curves
  pyplot.figure()
  colors = ['darkorange', 'blue', 'green', 'red', 'purple']  # Add more colors if needed
  for i, (falsePositiveRate, truePositiveRate, areaUnderCurve) in enumerate(rocsForPlotting):
    pyplot.plot(falsePositiveRate, truePositiveRate, color=colors[i % len(colors)], lw=2, label=f'ROC curve Exp: #{i + 1} (area = {areaUnderCurve:.2f})')
  pyplot.xlim([0, 1])
  pyplot.ylim([0, 1])
  pyplot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  pyplot.xlabel('False Positive Rate')
  pyplot.ylabel('True Positive Rate')
  pyplot.title('ROC for Custom Random Forests')
  pyplot.legend(loc="lower right")
  pyplot.show()

  # Display all res in a table
  resultsDataFrame = pd.DataFrame(res)
  print_table(resultsDataFrame)

run_and_plot_custom_random_forest(custom_rf_parameters)

class ModelTrainerSklearnPlots:
  def __init__(self, randomState, X, y,hasAnySpecialConditions=False):
      self.random_state = randomState
      self.models = {}
      self.X = X
      self.y = y

  def run_adaboost(self, n_estimators, learning_rate, testSize=0.2):
      XTrain, XTest, yTrain, yTest = train_test_split(self.X, self.y, test_size=testSize, random_state=self.random_state)
      classifier = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=self.random_state)
      classifier.fit(XTrain, yTrain)
      yPredection = classifier.predict(XTest)
      report = classification_report(yTest, yPredection, output_dict=True)
      self.models['ADABoost n_estimators=' + str(n_estimators) + " learning_rate=" + str(learning_rate)] = classifier
      return pd.DataFrame(report).transpose()

  def run_xgboost(self, n_estimators, learning_rate, testSize=0.2):
      XTrain, XTest, yTrain, yTest = train_test_split(self.X, self.y, test_size=testSize, random_state=self.random_state)
      classifier = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')
      classifier.fit(XTrain, yTrain)
      yPredection = classifier.predict(XTest)
      report = classification_report(yTest, yPredection, output_dict=True)
      self.models['XGBoost n_estimators=' + str(n_estimators) + " learning_rate=" + str(learning_rate)] = classifier
      return pd.DataFrame(report).transpose()

  def run_bagging(self, n_estimators, test_size=0.2):
      xTrain, xTest, yTrain, yTest = train_test_split(self.X, self.y, test_size=test_size, random_state=self.random_state)
      classification = BaggingClassifier(n_estimators=n_estimators, random_state=self.random_state)
      classification.fit(xTrain, yTrain)
      yPred = classification.predict(xTest)
      report = classification_report(yTest, yPred, output_dict=True)
      self.models['Bagging n_estimators=' + str(n_estimators)] = classification
      return pd.DataFrame(report).transpose()


  def plot_all_roc_curves(self):
    pyplot.figure(figsize=(10, 8))

    for modelName, classification in self.models.items():
        XTrain, XTest, yTrain, yTest = train_test_split(self.X, self.y, test_size=0.2, random_state=self.random_state)
        classification.fit(XTrain, yTrain)
        yProbs = classification.predict_proba(XTest)[:, 1]

        falsePositiveRate, truePositiveRate, _ = roc_curve(yTest, yProbs)
        areaUnderCurve = auc(falsePositiveRate, truePositiveRate)

        pyplot.plot(falsePositiveRate, truePositiveRate, lw=2, label=f'{modelName} (area = {areaUnderCurve:.2f})')

    pyplot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    pyplot.xlim([0, 1])
    pyplot.ylim([0, 1])
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('ROC Curves & Areas w/ Sklearn')
    pyplot.legend(loc="lower right")
    pyplot.show()

"""Sklearn Implementations. We are implementing Random Forests from scratch, but XGBoost, ADABoost, and Bagging by hand."""

X = dataFrame.iloc[:, 1:-1] # X feature columns
y = dataFrame.iloc[:, -1] # Last column (target_5yrs) for y

mt = ModelTrainerSklearnPlots(3, X, y)


results_df = pd.DataFrame(columns=['Experiment Number', 'Parameters (Model)', 'Results (Accuracy, Precision, Recall, F1-Score)'])
experiment_counter = 1
def add_results_to_df(report, model_name, params):
    global experiment_counter
    paramsString = ', '.join([f"{k}={v}" for k, v in params.items()])
    results_str = f"Accuracy: {round(report.loc['accuracy', 'precision'], 4)}, " \
                  f"Precision: {round(report.loc['weighted avg', 'precision'], 4)}, " \
                  f"Recall: {round(report.loc['weighted avg', 'recall'], 4)}, " \
                  f"F1-Score: {round(report.loc['weighted avg', 'f1-score'], 4)}"
    results_df.loc[experiment_counter] = [
        experiment_counter,
        f"{model_name} ({paramsString})",
        results_str
    ]
    experiment_counter += 1

for params in ada_parameters:
    ada_report = mt.run_adaboost(**params)
    add_results_to_df(ada_report, 'AdaBoost', params)

for params in xgb_parameters:
    xgb_report = mt.run_xgboost(**params)
    add_results_to_df(xgb_report, 'XGBoost', params)

for params in bag_parameters:
    bag_report = mt.run_bagging(**params)
    add_results_to_df(bag_report, 'Bagging', params)


#print("These experiments are done with a 80/20 train/test split.")
run_and_plot_custom_random_forest
print_table(results_df)
mt.plot_all_roc_curves()

