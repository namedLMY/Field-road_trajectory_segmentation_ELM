# import loaddata as lddata
import parameter as pm
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import hpelm


def train_and_evaluate(X_train, y_train, X_test, y_test):
    # 创建并训练ELM分类器
    hidden_units = pm.NUITS
    activation_function = "sigm"  # "sigm", "tanh", "lin"
    elm = hpelm.ELM(X_train.shape[1],
                    y_train.shape[1],
                    classification="c",
                    batch=pm.B)
    elm.add_neurons(hidden_units, activation_function)
    elm.train(X_train, y_train, "LOO")

    # 预测并计算准确率
    y_pred = elm.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    precision = precision_score(y_test_classes, y_pred_classes)
    recall = recall_score(y_test_classes, y_pred_classes)
    f1 = f1_score(y_test_classes, y_pred_classes)

    return precision, recall, f1, y_pred_classes, y_test_classes


# def visualize_results(X_test, y_pred_classes, y_test_classes):
#     # 将归一化后的X恢复为原始数据
#     X_test = lddata.scaler.inverse_transform(X_test)

#     # 绘制测试数据和预测结果
#     plt.scatter(X_test[y_test_classes == 1, 0],
#                 X_test[y_test_classes == 1, 1],
#                 c='g',
#                 marker='o',
#                 label='FIELD')
#     plt.scatter(X_test[y_test_classes == 0, 0],
#                 X_test[y_test_classes == 0, 1],
#                 c='b',
#                 marker='o',
#                 label='ROAD')
#     plt.scatter(X_test[y_test_classes != y_pred_classes, 0],
#                 X_test[y_test_classes != y_pred_classes, 1],
#                 c='r',
#                 marker='x',
#                 label='WRONG')
#     plt.xlabel('LND')
#     plt.ylabel('LAT')
#     plt.title('Test Data and Predicted Results')
#     plt.legend()
#     plt.show()
