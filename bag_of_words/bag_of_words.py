import os
import os.path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from pandas.testing import assert_frame_equal

# Importing data and checking if the csv files pre-exist

train_file = 'train_data.csv'
test_file = 'test_data.csv'
demo_file = 'demo.csv'

if(os.path.isfile(train_file) and os.path.isfile(test_file)):
	
	train_data = pd.read_csv(train_file)
	test_data = pd.read_csv(test_file)
else:
	data_frame = pd.read_csv('title_category_table.csv')
	data_frame = data_frame.sample(frac=0.3)
	categories = ['Travel','Science & Math','Romance','Sports & Outdoors','Cookbooks, Food & Wine']
	train_data, test_data = train_test_split(data_frame, train_size = 0.8, random_state = 44 )
	#train_data.to_csv( train_file )
	#test_data.to_csv( test_file )
	print("train and test csv created \n")

data = train_data

# Separating input from dataset 
data_as_string_array = data.iloc[:,1].values.astype('str');

data_string_list = data_as_string_array.tolist();

clean_list = data_string_list;

################ Count Vectorizer for Training Data ####################

# initializing a count vectorizer
count_vectorizer = CountVectorizer()

# fitting the vectorizer
count_vectorizer.fit(clean_list); 

#print(count_vectorizer.vocabulary_);

# transforming the strings into the vectors 
count_vector = count_vectorizer.transform(clean_list);

# converting it into an array to make it a viable input
count_data_array = count_vector.toarray();

# Training of data

print("Training the random forest...")

model_load_name = 'random_forest_model.pkl'

# Initialize a Random Forest classifier with 100 trees if a model file does not exist

if(os.path.isfile(model_load_name)):

	forest = joblib.load(model_load_name)
else:
	# Fit the forest to the training set, using the bag of words as 
	# features and the genre labels as the response variable
	#
	# This may take a few minutes to run
	
	forest = RandomForestClassifier(n_estimators = 100) 
	forest = forest.fit( count_data_array, data["Category"]);
	model_dump_name = 'random_forest_model.pkl'
	joblib.dump(forest, model_dump_name)

data = pd.read_csv(demo_file)
demo_data = data
#print("Equality is : ")
#print(assert_frame_equal(train_data, data))

data_as_string_array = data.iloc[:,1].values#.astype('str');

data_string_list = data_as_string_array.tolist();

clean_test_reviews = data_string_list;

#count_vectorizer = CountVectorizer()

#count_vectorizer.fit(clean_test_reviews); 

demo_data_features = count_vectorizer.transform(clean_test_reviews)
demo_data_features = demo_data_features.toarray()

# Use the random forest to make label predictions
result = forest.predict(demo_data_features)

features_file = 'second_last_layer_output.npy'

np.save(features_file, demo_data_features)

print("File written")

# Copy the results to a pandas dataframe with an "id" column and a "Category" column
output = pd.DataFrame( data={"Title":demo_data["Title"], "Category":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "output_demo_data_set.csv" )

# Testing accuracy
actual_output = demo_data["Category"]
print(accuracy_score(actual_output, result))
os.system('notify-send done!! ')
os.system('spd-say "Code khatam ho gyaaaa saaley"')
