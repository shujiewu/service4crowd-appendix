
import numpy as np
import sys
import json

if __name__ == "__main__":
    selectSize = int(sys.argv[1])
    seed = int(sys.argv[2])
    dataSetFile = str(str(sys.argv[3]))
    excludeIdFile = str(str(sys.argv[4]))
    dataSetName = str(str(sys.argv[5]))
    storage_dir = str(sys.argv[6])

    excludeIdSet = set()
    with open(excludeIdFile,'r') as f:
        excludeIds = json.load(f)
        for id in excludeIds:
           excludeIdSet.add(id)

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
    for i in range(len(imgs)):
        if imgs[perm[i]]['id'] not in excludeIdSet:
            select.append(imgs[perm[i]]['id'])
        if len(select)>=selectSize:
            break

    file_name = storage_dir+'/output'
    res = {}
    res['imageIdList']=select
    res['imageNum'] = len(select)

    with open(file_name,'w') as file_obj:
        json.dump(res,file_obj)
    print(res)





