import pymongo

myclient= pymongo.MongoClient("mongodb://127.0.0.1:27017/")

mydb = myclient["mydtabase"]
print(myclient.list_database_names())

dblist = myclient.list_database_names()
if "mydatabase" in dblist:
    print("the database exists.")
else:
    print("the database does not exists.")

#print(mydb.list_database_names())

mycol = mydb["user"]
#print(mydb.list_collection_names())
#A={"names":"test","age":22}
#mycol.insert_one(A)

# A=[{"names":"fuck","age":22,"_id":1},{"names":"off","age":44,"_id":2}]
# mycol.insert_many(A)

#find_one => afficher le premier element du coolection
# x = mycol.find_one()
# print(x)

# #find() => afficher tous les elements du collection

# for i in mycol.find():
#     print(i)

# #afficher tous les elements sauf qui ont un id=0
# for i in mycol.find({},{"_id":0}):
#     print(i)


# for i in mycol.find({"names":"test"},{"_id":0}):
#     print(i)

# for i in mycol.find({"age":{"$gt":18}}):
#     print(i)

# # mycol.delete_one({"names":"test"})

# mycol.delete_many({"age":{"$gt":44}})

# mycol.delete_many({})
# mycol.drop()

# myquery = {"names":"test"}
# mynewvalue={"$set":{"names":"test0"}}
# mycol.update_one(myquery,mynewvalue)

try:
    l=[1,2,3]
    print(l[5])
except IndexError as e:
    print("il ya un error")
print("hello")



