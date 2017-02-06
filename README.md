# mushroomobser-dataset
Mushrooms images dataset collected from [http://mushroomobserver.org](http://mushroomobserver.org)

# About the Dataset

http://mushroomobserver.org exists since 2006, since then mushroom enthusiasts contribute on a daily basis to the collection of mushroom observations on the website. The site hast approximately 10.000 users that contributed in total approximately 250.000 observations of mushrooms. Each observations counts 1-5 images. 

## Structure of the Dataset
The dataset consists of pictures of mushrooms, the pictures are sorted by year and by label. Label can be species label but also more general label in the taxonomy of the mushroom like kingdom, phylum, class, family, order or genus. 
In addition to the full dataset, a "clean" dataset hast been created, the clean dataset contains only the thumbnail image of every observation.

The most recent year is always used as test dataset, all the other years as training dataset.

The dataset collected can be downloaded [here](https://www.dropbox.com/sh/m1o91dwd1nto6w0/AABuDQVJWTq04lL_yaF_G2MFa?dl=0).
Additionally json files containing additional information about each image can be downloaded from the above link as well. The json files contain a list of dictionaries, each dictionary contains following information about the image, if image[thumbnail]=1 the image belongs to the clean dataset:
```python
{'date': '2006-05-21 07:17:22',
 'gbif_info': {'canonicalName': 'Xerocomells dryophils',
  'class': 'Agaricomycetes',
  'classKey': 186,
  'confidence': 98,
  'family': 'Boletaceae',
  'familyKey': 8789,
  'gens': 'Xerocomells',
  'gensKey': 8184844,
  'kingdom': 'Fngi',
  'kingdomKey': 5,
  'matchType': 'EXACT',
  'order': 'Boletales',
  'orderKey': 1063,
  'phylm': 'Basidiomycota',
  'phylmKey': 34,
  'rank': 'SPECIES',
  'scientificName': 'Xerocomells dryophils (Thiers) N. Siegel, C.F. Schwarz & J.L. Frank, 2014',
  'species': 'Xerocomells dryophils',
  'speciesKey': 7574003,
  'stats': 'ACCEPTED',
  'synonym': False,
  'sageKey': 7574003},
 'image_id': 11,
 'image_rl': 'http://mshroomobserver.org/images/320/11',
 'label': 'Xerocomells dryophils',
 'location': 38,
 'observation': 10,
 'thmbnail': 1,
 'user': 1}
```

The file mushroom_taxonomy.pdf shows an overview of the taxonomy of the mushroom dataset.

## Scrape newest year for test
To scrape the images from most recent year from [http://mushroomobserver.org](http://mushroomobserver.org) you can run scrape_images_of_year.py
```bash
python download_images_of_year.py year destination_folder
```
you may stop the script with ctrl+C as soon as it starts scraping exclusively observations that are older then the desired year. The script creates a json file with the image information.

## Create species dataset 

To create a dataset containing n classes of only mushroom species from the training set the script create_data_set.py can be used. Arguments are number of classes wanted, path to training dataset, path to validation dataset.
```bash
python create_data_set.py 10 /Volumes/MO/Trainingset /Volumes/MO/Validationset
```

# Performance evaluation
The TensorFlow-Slim image classification library was used, for installation instructions see [here](https://github.com/tensorflow/models/tree/master/slim)
The code for performance evaluation is stored in the slim folder of this repository. 

### Create tensorflow dataset
to create tensorflow dataset:
```bash
TRAIN_DIR=/Volumes/MO/DATA/TRAIN_10
TEST_DIR=/Volumes/MO/DATA/VALIDATE_10

python download_and_convert_data.py \
    --dataset_name=mushrooms \
    --train_dir=${TRAIN_DIR} \
    --test_dir=${TEST_DIR}
```
The tensorflow dataset is stored in slim/tf_data

### Train network

A pre-trained inception_v3 network pre-trained on Imagenet was used. The network is stored in slim/inception_v3. To finetune on the mushroom dataset:

```bash
DATASET_DIR=./tf_data
TRAIN_DIR=./train_models
CHECKPOINT_PATH=./inception_v3/inception_v3.ckpt

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=mushrooms \
    --dataset_split_name=train \
    --model_name=inception_v3 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --max_number_of_steps=100000 \
  	--batch_size=32 \
  	--learning_rate=0.001 \
  	--learning_rate_decay_type=exponential \
  	--save_interval_secs=3600 \
  	--save_summaries_secs=3600 \
  	--log_every_n_steps=1000 \
  	--optimizer=rmsprop \
 	--weight_decay=0.00004
``` 


### Evaluate network
To evaluate the network performance:
```bash
DATASET_DIR=./tf_data
CHECKPOINT_FILE=./train_models_kopie/model.ckpt

python eval_image_classifier.py \
	--alsologtostderr \
	--checkpoint_path=${CHECKPOINT_FILE} \
	--dataset_dir=${DATASET_DIR} \
	--dataset_name=mushrooms \
	--dataset_split_name=validation \
	--model_name=inception_v3
```
