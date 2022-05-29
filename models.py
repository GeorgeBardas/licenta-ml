import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM


def load_data(file_name):
    return pd.read_csv(file_name)


model_scores = {}


def tts(data):
    """
    Impartirea datelor in date de antrenare si date de test
    Datele de test sunt formate din ultimele 12 luni de informatii
    """
    data = data.drop(['sales', 'date'], axis=1)
    train, test = data[0:-12].values, data[-12:].values

    return train, test


def scale_data(train_set, test_set):
    """
    Scalare folosing MinMaxScaler
    Separarea informatiilor in x_train, y_train, x_test, y_test
    """

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)

    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set)

    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)

    x_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1].ravel()
    x_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()

    return x_train, y_train, x_test, y_test, scaler


def undo_scaling(y_pred, x_test, scaler_obj, lstm=False):
    """
    Rescalarea datelor pentru vizualizare si comparare
    """

    y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)

    if not lstm:
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    pred_test_set = []
    for index in range(0, len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[index], x_test[index]],
                                            axis=1))

    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0],
                                          pred_test_set.shape[2])

    pred_test_set_inverted = scaler_obj.inverse_transform(pred_test_set)

    return pred_test_set_inverted


def predict_df(unscaled_predictions, original_df):
    """
    Generare informatiilor ce arata predictia lunara a vanzarilor
    """
    result_list = []
    sales_dates = list(original_df[-13:].date)
    act_sales = list(original_df[-13:].sales)

    for index in range(0, len(unscaled_predictions)):
        result_dict = {}
        result_dict['pred_value'] = int(unscaled_predictions[index][0] +
                                        act_sales[index])
        result_dict['date'] = sales_dates[index + 1]
        result_list.append(result_dict)

    df_result = pd.DataFrame(result_list)

    return df_result


def get_scores(unscaled_df, original_df, model_name):
    """
    Printare root mean squared error, mean absolute error, R2 scores pentru fiecare model
    """
    rmse = np.sqrt(mean_squared_error(original_df.sales[-12:], unscaled_df.pred_value[-12:]))
    mae = mean_absolute_error(original_df.sales[-12:], unscaled_df.pred_value[-12:])
    r2 = r2_score(original_df.sales[-12:], unscaled_df.pred_value[-12:])
    model_scores[model_name] = [rmse, mae, r2]

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")


def plot_results(results, original_df, model_name):
    """
    Generare grafice peste cele originale pentru vizualizare
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.lineplot(original_df.date, original_df.sales, data=original_df, ax=ax,
                 label='Original', color='mediumblue')
    sns.lineplot(results.date, results.pred_value, data=results, ax=ax,
                 label='Predictie', color='red')
    ax.set(xlabel="Data",
           ylabel="Vanzari",
           title=f"{model_name} Predictie vanzari")
    ax.legend()
    sns.despine()

    plt.savefig(f'/Users/georgebardas/Documents/Projects/python/licenta-ml/model_output/{model_name}_predictie.png')


def regressive_model(train_data, test_data, model, model_name):
    """
    Rularea modelelor regresive in SKlearn framework
    Predictia
    Printare rezultate
    Generare grafice
    """

    x_train, y_train, x_test, y_test, scaler_object = scale_data(train_data,
                                                                 test_data)
    mod = model
    mod.fit(x_train, y_train)
    predictions = mod.predict(x_test)

    original_df = load_data('/Users/georgebardas/Documents/Projects/python/licenta-ml/data/monthly_data.csv')
    unscaled = undo_scaling(predictions, x_test, scaler_object)
    unscaled_df = predict_df(unscaled, original_df)

    get_scores(unscaled_df, original_df, model_name)
    plot_results(unscaled_df, original_df, model_name)


def lstm_model(train_data, test_data):
    """
    Rulare LSTM(2 dense layers)
    Printare rezultate
    Generare grafice
    """

    x_train, y_train, x_test, y_test, scaler_object = scale_data(train_data, test_data)

    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    model = Sequential([
        LSTM(4, batch_input_shape=(1, x_train.shape[1], x_train.shape[2])),
        Dense(1),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(x_train, y_train, epochs=200, batch_size=1, verbose=1, shuffle=False)
    predictions = model.predict(x_test, batch_size=1)

    original_df = load_data('/Users/georgebardas/Documents/Projects/python/licenta-ml/data/monthly_data.csv')
    unscaled = undo_scaling(predictions, x_test, scaler_object, lstm=True)
    unscaled_df = predict_df(unscaled, original_df)

    get_scores(unscaled_df, original_df, 'LSTM')
    plot_results(unscaled_df, original_df, 'LSTM')


def main():
    # Modele regresie
    model_df = load_data('/Users/georgebardas/Documents/Projects/python/licenta-ml/data/model_df.csv')
    train, test = tts(model_df)

    # Sklearn
    regressive_model(train, test, LinearRegression(), 'LinearRegression')

    # Keras
    lstm_model(train, test)


main()

# Salvarea scorurilor modelelor pentru comparare in results.py
pickle.dump(model_scores, open("model_scores.p", "wb"))
