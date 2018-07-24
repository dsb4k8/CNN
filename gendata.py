import pandas as pd 
import numpy as np
import cPickle #for unzipping of data
#for data conversion
import graphlab as gl 


def unpickle(file):
	fo = open(file, "rb")
	dict = cPickle.load(fo)
	fo.close()
	return dict

batch1 = unpickle("data/data_batch_1")
batch2 = unpickle("data/data_batch_2")
batch3 = unpickle("data/data_batch_3")
batch4 = unpickle("data/data_batch_4")
batch5 = unpickle("data/data_batch_5")
batch_test = unpickle("data/test_batch")

def get_dataframe(batch):
	df = pd.DataFrame(batch["data"])
	df["image"] = df.as_matrix().tolist()
	df.drop(range(3072), axis = 1, inplace = True)
	df["label"] = batch["labels"]
	return df

train = pd.concat([get_dataframe(batch1),get_dataframe(batch2),get_dataframe(batch3),get_dataframe(batch4),get_dataframe(batch5)], ignore_index = True)
test = get_dataframe(batch_test)

print train.head()
print train.shape, test.shape

gltrain = gl.SFrame(train)
gltest = gl.SFrame(test)
model = gl.neuralnet_classifier.create(gltrain, target = "label", validation_set = None)
model.evaluate(gltest)


gltrain['glimage'] = gl.SArray(gltrain['image']).pixel_array_to_image(32,32,3, allow_rounding = True)
gltest['glimage'] = gl.SArray(gltest['image']).pixel_array_to_image(32,32,3, allow_rounding = True)

gltrain.remove_column("image")
gltest.remove_column("image")
gltrain.head()

gltrain['image'] = gl.image_analysis.resize(gltrain['glimage'], 256,256,3)
gltest['image'] = gl.image_analysis.resize(gltest['glimage'], 256,256,3)

gltrain.remove_column('glimage')
gltest.remove_column('glimage')
gltrain.head()

pretrained_model = gl.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')
gltrain['features'] = pretrained_model.extract_features(gltrain)
gltest['features'] = pretrained_model.extract_features(gltest)
gltrain.head()