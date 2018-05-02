import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

data_frame = pd.read_csv('title_category_table.csv')

#### Cleaning the data is left ######
#print(np.where(data_frame.applymap( lambda x: x==' ')))

categories = ['Travel','Science & Math','Romance','Sports & Outdoors','Cookbooks, Food & Wine']

train_data, test_data = train_test_split(data_frame, train_size = 0.8, random_state = 44 )

#for i in categories:
#	
#	data_segment = train_data.loc[data_frame['Category']==i]
#	print(i)	
#	print("\n")
#	print(data_segment[:5])
#	print("\n")

data = train_data

data_as_string_array = data.iloc[:,1].values.astype('str');

data_string_list = data_as_string_array.tolist();

clean_list = data_string_list;

################ Count Vectorizer ####################

count_vectorizer = CountVectorizer()

count_vectorizer.fit(clean_list); 

#print(count_vectorizer.vocabulary_);

count_vector = count_vectorizer.transform(clean_list);

#print(count_vector.toarray());
count_data_array = count_vector.toarray();

## Training of data ###
print("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( count_data_array, data["Category"]);

####### Test Data splitting and accuracy checking #########

data = test_data

data_as_string_array = data.iloc[:,1].values#.astype('U');

data_string_list = data_as_string_array.tolist();

clean_test_reviews = data_string_list;

test_data_features = count_vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"Title":test_data["Title"], "Category":result} )

print(output)

# Use pandas to write the comma-separated output file
#output.to_csv( "output.csv", index=False, quoting=3 )
output.to_csv( "output.csv")
