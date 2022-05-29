import pandas as pd


def load_data():
    return pd.read_csv("datasets/train.csv")


def monthly_sales(data):
    monthly_data = data.copy()

    # Stergere zi din coloana datei
    monthly_data.date = monthly_data.date.apply(lambda x: str(x)[:-3])

    # Suma vanzari/luna
    monthly_data = monthly_data.groupby('date')['sales'].sum().reset_index()
    monthly_data.date = pd.to_datetime(monthly_data.date)

    monthly_data.to_csv('/Users/georgebardas/Documents/Projects/python/licenta-ml/data/monthly_data.csv')

    return monthly_data


def get_diff(data):
    data['sales_diff'] = data.sales.diff()
    data = data.dropna()

    data.to_csv('/Users/georgebardas/Documents/Projects/python/licenta-ml/data/stationary_df.csv')

    return data


def generate_supervised(data):
    supervised_df = data.copy()

    # crare coloana pentru fiecare lag
    for i in range(1, 13):
        col_name = 'lag_' + str(i)
        supervised_df[col_name] = supervised_df['sales_diff'].shift(i)

    # stergere valori nule
    supervised_df = supervised_df.dropna().reset_index(drop=True)

    supervised_df.to_csv('/Users/georgebardas/Documents/Projects/python/licenta-ml/data/model_df.csv', index=False)


def main():
    """
    Incarcare date de antrenare
    Generare date lunare
    Calculare diferente si generare date stationare
    Creare fisiere csv pentru regresie
    """
    sales_data = load_data()
    monthly_df = monthly_sales(sales_data)
    stationary_df = get_diff(monthly_df)

    generate_supervised(stationary_df)


main()
