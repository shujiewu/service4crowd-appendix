
import numpy as np
import sys
import json

if __name__ == "__main__":
    selectSize = int(sys.argv[1])
    dataSetFile = str(str(sys.argv[2]))
    seed = int(sys.argv[3])
    storage_dir = str(sys.argv[4])

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
        select.append(imgs[perm[i]])
    # selectStorageFile = storage_dir + '/selectResult'
    # with open(dataSetFile, 'wt') as f:
    #     json.dump(select,f)
    file_name = storage_dir+'/output'
    res = {}
    res['selectResult']=select
    with open(file_name,'w') as file_obj:
        json.dump(res,file_obj)
    print(res)





