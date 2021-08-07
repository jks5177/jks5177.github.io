# 라벨 전파 훈련

저번에 YOLO를 수정해서 영상에서 특정 객체의 이미지를 뽑아내는 과정을 보았습니다.

영상을 실시간으로 Detection 해주는 YOLO 소스코드를 수정해서 bounding box 기준으로 잘라 줬습니다.

이번에는 잘라진 이미지들을 이용해서 라벨 전파 훈련을 진행할 것입니다.

라벨 전파 훈련은 Semi-Supervised Learining으로 준지도 학습에 해당됩니다.

준지도 학습이란 라벨이 없는 데이터가 라벨이 있는 데이터보다 많을때 진행하는 학습 방법입니다. 라벨 전파를 통해 라벨이 없는 데이터들에게 라벨링을 자동으로 해주는 과정입니다.

우선 아래 코드들을 보면서 진행하도록 하겠습니다.

코랩을 이용해서 진행하였으며 데이터는 저번에 잘라둔 이미지를 구글 드라이브에 넣어 놓고 코랩에 mount해서 진행했습니다.

즉, 데이터 파일의 경로를 잘 수정 후 사용해 주셔야합니다.

## 데이터 로드

우선 데이터들을 불러 오겠습니다.


```python
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
```


```python
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
```


```python
!cp -r gdrive/MyDrive/CodingAlzi/Cow/* image/ #ID가 있는 소들
!cp -r gdrive/MyDrive/CodingAlzi/cut_video /content/ #영상에서 자른 증가된 소 이미지
```


```python
#라벨이 있는 파일들을 경로 명으로 저장
label_path = "/content/image"
os.chdir(label_path)

label_cows = []

with os.scandir(label_path) as label_files :
  for label_file in label_files :
    if label_file.name.endswith('.jpg') :
      label_cows.append(label_file.name)

#라벨이 없는 파일들을 경로 명으로 저장
nolabel_path = "/content/cut_video"
os.chdir(nolabel_path)

nolabel_cows = []

with os.scandir(nolabel_path) as nolabel_folders :
  for nolabel_folder in nolabel_folders :
    with os.scandir(nolabel_folder) as nolabel_files :
      for nolabel_file in nolabel_files :
        if nolabel_file.name.endswith('.jpg') :
          nolabel_cows.append(nolabel_folder.name + "/" + nolabel_file.name)
```


```python
y_label = []
for label_cow in label_cows :
  name = label_cow.split("_")[1]
  y_label.append(int(name))
```

데이터들을 다 불러왔습니다.

## 특성 추출

라벨 전파를 위해 특성을 추출을 하겠습니다.

라벨 전파는 같은 군집에 있는 데이터끼리 같은 라벨을 갖기때문에 특성을 추출한 후 데이터들을 군집화 해주어야 합니다.

특성을 추출하기 위해 VGG16 모델을 사용했습니다.


```python
# 특성을 추출할 모델을 부르고 이미지 전처리 후 특성 추출

#load model
model = VGG16()
# remove the output layer
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

def extract_features(file, model) :
  # 이미지를 224x224 크기로 불러옴니다.
  img = load_img(file, target_size=(224,224))
  # 이미지를 numpy 배열로 변경해줍니다.
  img = np.array(img)
  # 모델의 입력 층에 맞게 4차원으로 변경해줍니다.
  reshaped_img = img.reshape(1, 224, 224, 3)
  # 모델에 맞게 이미지를 전처리 해줍니다.
  imgx = preprocess_input(reshaped_img)
  # 특성 벡터를 추출합니다.
  features = model.predict(imgx, use_multiprocessing=True)
  return features
```

위에서 이미지의 위치만 리스트에 넣었기 때문에 리스트에 있는 이미지들을 불러와서 특성을 뽑은 후 저장하도록 하겠습니다.


```python
label_data = {}
os.chdir(label_path)
p = "/content/image/"

# lop through each image in the dataset
for cow in label_cows :
  # try to extract the geatures and update the dictionary
  try :
    feat = extract_features(cow, model)
    label_data[cow] = feat
  # if something ails, save the extracted features as a pickle file (optional)
  except :
    with open(p, 'wb') as file:
      pickle.dump(label_data, file)

nolabel_data = {}
os.chdir(nolabel_path)
p = "/content/cut_vidoe"

# lop through each image in the dataset
for cow in nolabel_cows :
  # try to extract the geatures and update the dictionary
  try :
    feat = extract_features(cow, model)
    nolabel_data[cow] = feat
  # if something ails, save the extracted features as a pickle file (optional)
  except :
    with open(p, 'wb') as file:
      pickle.dump(nolabel_data, file)
```

## 라벨 전파 훈련

이제 라벨 전파 훈련을 진행하도록 하겠습니다.


```python
# evaluate logistic regression fit on label spreading for semi-supervised learning
from numpy import concatenate
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelSpreading
from sklearn.linear_model import LogisticRegression
```

우선 라벨이 있는 데이터들과 라벨이 없는 데이터들을 합쳐줍니다.


```python
X_mixed = concatenate((label_feat, nolabel_feat))
nolabel = [-1 for _ in range(len(nolabel_feat))]
y_mixed = concatenate((y_label, nolabel))
print(X_mixed.shape)
print(y_mixed.shape)
```

그 후 LabelSpreading() 함수를 사용해서 라벨 전파를 실행하였습니다.


```python
#LabelSpreading() 함수를 이용해서 라벨 전파를 실행하였습니다.
model = LabelSpreading()
model.fit(x, y_mixed)
tran_labels = model.transduction_
```

아래 코드는 라벨 전파가 얼마나 잘되었는지 확인하기 위해 dataframe 만들어 줬습니다.


```python
# 인덱스 값으로 label 값이 들어가게 했습니다.
dic = {}
dic[1] = tran_labels
df = pd.DataFrame(data=dic, index=name)
df = df.sort_index()
df.index

# 라벨 값을 이용해서 dataframe을 생성합니다.
frame_dic = {}
for x in df.index :
  try:
    frame_name = x.split("/")[0] + "/" + x.split("/")[1].split("-")[0]
    if frame_name not in frame_dic.keys() :
      frame_dic[frame_name] = str(int(df.loc[[x]].values)) + "!" + '-'.join(x.split("/")[1].split("-")[1:])
      frame_dic[frame_name] = frame_dic[frame_name].values.tolist()
    else :
      frame_dic[frame_name] = np.append(frame_dic[frame_name], str(int(df.loc[[x]].values))+ "!" + '-'.join(x.split("/")[1].split("-")[1:]))
      frame_dic[frame_name] = sorted(frame_dic[frame_name])
  except:
    pass
```

## 데이터 저장

라벨 전파 훈련은 끝났습니다.

하지만 PCA의 값이 변경됨에 따라 데이터들이 어떻게 변형 되지는 확인하기 위해 PCA값을 다양하게 설정했습니다.


```python
dic = {}
for i in range(1, 4097) :
  pca = PCA(n_components=1, random_state=22)
  pca.fit(X_mixed)
  x = pca.transform(X_mixed)

  model = LabelSpreading()
  model.fit(x, y_mixed)
  tran_labels = model.transduction_

  dic[i] = tran_labels
```


```python
df = pd.DataFrame(data=dic, index=name)
df.to_csv('/content/result.csv')
```

데이터가 너무 많아 Dataframe을 생성한 후 csv 파일로 변환해서 로컬로 확인했습니다.

그 결과 pca 값은 라벨 전파에 큰 차이를 주지 못했습니다.

## 결과

2번의 포스팅으로 YOLO를 이용한 영상에서 이미지 객체 추출과 라벨 전파 훈련을 작성해 봤습니다.

저는 특정 객체에 모든 라벨을 가지고 있지 못했고 이미지도 다양한 방향에서 촬영하지 못해서 성능이 좋지 못한 데이터셋을 생성했습니다.

이와 같은 방향으로 적은 이미지로 다양한 이미지를 얻을 수 있지만 라벨이 달린 이미지는 꼭 Unique 값 만큼 있어야합니다. 또한 다양한 방향의 사진이 있을 수록 더 정확한 데이터셋을 만들 수 있습니다.
