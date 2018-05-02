import pandas as pd


data = pd.read_csv('book32-listing.csv');

new_data = data.iloc[:,[3,6]];

new_data.set_index('Category', inplace=True);

needed_data_labels = new_data.loc[['Travel','Science & Math','Romance','Sports & Outdoors','Computer & Technology','Cookbooks, Food & Wine']];

needed_data_labels.to_csv('test_data.csv');
