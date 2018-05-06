import pandas as pd

# Separating the favourable categories from the dataset to form a subset

data = pd.read_csv('book32-listing.csv');

new_data = data.iloc[:,[3,6]];

new_data.set_index('Category', inplace=True);

needed_data_labels = new_data.loc[['Travel','Science & Math','Romance','Sports & Outdoors','Cookbooks, Food & Wine']];

needed_data_labels.to_csv('title_category_table.csv');
