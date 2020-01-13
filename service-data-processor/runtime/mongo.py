import pymongo
from gridfs import GridFS
from bson.objectid import ObjectId


class MongoGridFS(object):
    '''
    classdocs
    '''
    dbURL = "mongodb://crowd:1706223@10.1.1.63:27017/CrowdData"
    def __init__(self, params):
        '''
        Constructor
        '''
        # 上传文件

    def upLoadFile(self, file_coll, file_path, file_name, task_id):
        client = pymongo.MongoClient(self.dbURL)
        db = client["CrowdData"]
        filter_condition = {"fileName": file_name, "taskId": task_id, "filePath":file_path}
        gridfs_col = GridFS(db, collection=file_coll)
        file_ = "0"
        if gridfs_col.exists(filter_condition):
            print('已经存在该文件')
        else:
            with open(file_path, 'rb') as file_r:
                file_data = file_r.read()
                file_ = gridfs_col.put(data=file_data, **filter_condition)  # 上传到gridfs
                print(file_)
        return file_

    def downLoadFile(self, file_coll, file_name, out_name, ver):
        client = pymongo.MongoClient(self.dbURL)

        db = client["store"]

        gridfs_col = GridFS(db, collection=file_coll)

        file_data = gridfs_col.get_version(filename=file_name, version=ver).read()

        with open(out_name, 'wb') as file_w:
            file_w.write(file_data)

    # 按文件_Id获取文档
    def downLoadFilebyID(self, file_coll, _id, out_name):
        client = pymongo.MongoClient(self.dbURL)
        db = client["CrowdData"]
        gridfs_col = GridFS(db, collection=file_coll)
        O_Id = ObjectId(_id)
        gf = gridfs_col.get(file_id=O_Id)
        file_data = gf.read()
        with open(out_name, 'wb') as file_w:
            file_w.write(file_data)
        return gf.filename

if __name__ == '__main__':
    a = MongoGridFS("")
    # a.upLoadFile("pdf","MongoGridFS.py","")
    # a.downLoadFile("pdf","MongoGridFS.py","out2.p",2)
    ll = a.downLoadFilebyID("fs", "5de733b8adbb4a05c434ca71", "D://w.py")