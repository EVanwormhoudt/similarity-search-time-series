import time
from datetime import timedelta, datetime
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from index import SKLearnJenaIndex
from extractors import JenaPeriodExtractor

csv_path = 'data/jena_climate_2009_2016.csv'
jena_data = pd.read_csv(csv_path)

# extract data by taking value by hour instead of ten minutes
jena_data = jena_data[5::6]

jena_data['Date Time'] = pd.to_datetime(jena_data['Date Time'], format='%d.%m.%Y %H:%M:%S')

# min max - normalization
date_col = 'Date Time'
columns = jena_data.columns
means = {col: jena_data[col].mean() for col in columns}
stds = {col: jena_data[col].std() for col in columns}
for col in columns:
    if col != date_col:
        print('Normalize %s' % col)
        # min-max normalization
        # jena_data[col] = (jena_data[col] - jena_data[col].min()) \
        #                 / (jena_data[col].max() - jena_data[col].min())

        # standard normalization
        jena_data[col] = (jena_data[col] - means[col]) / stds[col]
jena_data.set_index('Date Time', inplace=True)

# split data into two subsets: indexed data and query data
n = len(jena_data)
indexed_data = jena_data[:int(n*0.9)]
queries_data = jena_data[int(n*0.9):]

def prepare_data(dataframe):
    items_dict = {}
    for row in dataframe.values.tolist():
        key = str(row[0])
        values = row[1:]
        items_dict[key] = values
    return items_dict

def prepare_data_by_period(dataframe, days):
    items_dict = {}
    df_by_day = dataframe.resample('D').mean()
    for n, i in enumerate(range(0, len(df_by_day), days)):
        period_data = df_by_day[i:i + days]
        if len(period_data) == days:
            year = period_data.iloc[0].name.year
            key = str(year) + "-p" + str(n)
            items_dict[key] = period_data
    return items_dict


items = prepare_data_by_period(indexed_data, days=7)
queries = prepare_data_by_period(queries_data, days=7)

jena_extractor = JenaPeriodExtractor(features=['T (degC)', 'p (mbar)'])
jena_index = SKLearnJenaIndex("sklearn_jena", jena_extractor, algorithm='brute')
# jena_index = LSHJenaIndex("lsh_jena", jena_extractor)
# jena_index = BruteForceJenaIndex("bf_jena", jena_extractor, dist='euclidean')
jena_index.index(items)

# jena_index.save("jena/data")
# jena_index.load("jena/data")

counter = Counter()
for period, query in queries.items():
    print("search for query:%s - %s" % (period, query))
    start_search = time.monotonic()
    periods, distances = jena_index.search(query, k=2)
    end_search = time.monotonic()
    search_time = end_search - start_search
    print("search time:%s" % timedelta(seconds=search_time))
    print("search results:")
    for period, dist in zip(periods[0], distances[0]):
       print("period:%s" % period)
       print("distances: %s" % dist)
       features = jena_index.features_at(period)
       print("features: %s" % features)

    # Plot both time series on the same graph
    plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
    index = list(range(0, 14))
    query_data = jena_extractor.extract(query)
    plt.plot(index, jena_index.features_at(periods[0][0]), marker='o', linestyle='-', label='result 1')
    plt.plot(index, jena_index.features_at(periods[0][1]), marker='+', linestyle='-', label='result 2')
    plt.plot(index, query_data, marker='x', linestyle='-', label='query')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.title('Time Series Comparison')
    plt.legend()  # Display legend to distinguish the two series
    plt.show()


# reduction
jena_data = jena_data.resample('D').mean()

# kNN to predict values
nb_query = len(queries_data)
true_values = []
predicted_values = []
for i in range(0, nb_query):
    query_row = queries_data.iloc[i]
    query_array = query_row[1:].values
    results = jena_index.search(query_array.tolist())
    best_score, best_date, best_idx = results[0]
    true_values.append(query_array)
    best_array = indexed_data[best_idx]
    predicted_values.append(best_array)

def plot_results(predicted_values, true_values):
    x_list = range(0, len(predicted_values))
    plt.plot(x_list[:200], predicted_values[:200], label='forecast')
    plt.plot(x_list[:200], true_values[:200], label='true')
    plt.legend()
    plt.show()

