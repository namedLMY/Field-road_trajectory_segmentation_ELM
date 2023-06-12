import loaddata as lddata
import parameter as pm
import numpy as np
import ELM
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


def main():
    X, y, y_one_hot = lddata.X, lddata.y, lddata.y_one_hot

    # 设置交叉验证
    cv = StratifiedKFold(n_splits=pm.N)

    # 记录准确率
    precisions = []
    recalls = []
    f1s = []
    i = 0

    # 遍历交叉验证的每个分组
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_one_hot[train_index], y_one_hot[test_index]

        precision, recall, f1, y_pred_classes, y_test_classes = ELM.train_and_evaluate(
            X_train, y_train, X_test, y_test)

        # 将归一化后的X恢复为原始数据
        X_test = lddata.scaler.inverse_transform(X_test)

        plt.figure(num='Test Data and Predicted Results', figsize=(18, 8))
        plt.subplot(2, 3, i + 1)

        # 绘制测试数据和预测结果
        plt.scatter(X_test[y_test_classes == 1, 0],
                    X_test[y_test_classes == 1, 1],
                    c='g',
                    marker='o',
                    label='FIELD')
        plt.scatter(X_test[y_test_classes == 0, 0],
                    X_test[y_test_classes == 0, 1],
                    c='b',
                    marker='o',
                    label='ROAD')
        plt.scatter(X_test[y_test_classes != y_pred_classes, 0],
                    X_test[y_test_classes != y_pred_classes, 1],
                    c='r',
                    marker='x',
                    label='WRONG')
        plt.xlabel('LND')
        plt.ylabel('LAT')
        plt.legend()
        i += 1

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    print("Precision:", np.mean(precisions))
    print("Recall:", np.mean(recalls))
    print("F1 score:", np.mean(f1s))

    plt.show()


if __name__ == '__main__':
    main()
