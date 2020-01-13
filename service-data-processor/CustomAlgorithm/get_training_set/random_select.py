
import numpy as np
import sys
import json

if __name__ == "__main__":
    selectSize = int(sys.argv[1])
    seed = int(sys.argv[2])
    dataSetFile = str(str(sys.argv[3]))
    dataSetName = str(str(sys.argv[4]))
    storage_dir = str(sys.argv[5])

    np.random.seed(seed)
    with open(dataSetFile, 'r') as f:
        dataset = json.load(f)
    imgs = []
    for img in dataset['images']:
        imgs.append(img)
    perm = np.random.permutation(np.arange(len(imgs)))
    if selectSize> len(imgs):
        raise Exception("selectSize is too large")
    select = []
    for i in range(selectSize):
        select.append(imgs[perm[i]]['id'])

    categoryMap = {}
    for category in dataset['categories']:
        categoryMap[category['id']]=category

    imgToAnn = {}
    for annotation in dataset['annotations']:
        imageId = annotation['image_id']
        categoryId = annotation['category_id']
        bbox = annotation['bbox']
        box={}
        box['x']=bbox[0]
        box['y']=bbox[1]
        box['w']=bbox[2]
        box['h']=bbox[3]
        box['score']=1.0
        classification={}
        classification['id']=categoryId
        classification['value'] = categoryMap[categoryId]['name']
        tag={}
        tag['box']=box
        tag['classification']=classification
        if imageId not in imgToAnn:
            imgToAnn[imageId]=[]
        imgToAnn[imageId].append(tag)

    trainingItemList =[]
    for img in select:
        trainingItem = {}
        trainingItem['dataSetName'] = dataSetName
        trainingItem['imageId'] = img
        trainingItem['tagList'] = imgToAnn[img]
        trainingItemList.append(trainingItem)

    file_name = storage_dir+'/output'
    res = {}
    res['imageIdList']=select
    res['trainingItemList']=trainingItemList

    with open(file_name,'w') as file_obj:
        json.dump(res,file_obj)
    print(res)





