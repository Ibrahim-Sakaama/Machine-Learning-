#in cmd
#mongod --dbpath data/db


import pymongo


#1) creation d'une base de donne
#1-- il faut d'abord creer un object MongoClient
myclient = pymongo.MongoClient("mongodb://localhost:27017/")

#le nom de la base de donnee
mydb = myclient["mydatabase2"]


#pour verifier si la bd exist or not

# db = myclient.list_database_names()
# if "mydatabase" in dblist:
#     print("the database exists")



mycol = mydb["customers"]

#inserer dans la collection

mydict = {"name":"John", "adresse":"Highway 37"}
#x = mycol.insert_one(mydict)
#return the id of the object inserted
#print(x.inserted_id)

mylist = [
    {"_id":1,"name":"Ibrahim", "adressse":"Sousse"},
    {"_id":2,"name":"Hana", "adressse":"KK"},
    {"_id":3,"name":"Ali", "adressse":"Hammam Sousse"},
    {"_id":4,"name":"Mohamed", "adressse":"KS"}
]


#x = mycol.insert_many(mylist)

#print list of the _id values of the inserted documents
#print(x.inserted_ids)

#find_one() ===> returns the first data/element of the collection
x = mycol.find_one()
#print(x)

#find() ===> returns all data in the collection
for x in mycol.find():
    print(x)

#finding some specific values

for x in mycol.find({},{"_id":0,"name":1,"adressse":1}):
    print(x)

#finding the adress that begins with the letter S or greater
myquery = {"adressse":{"$gt":"S"}}
mydoc = mycol.find(myquery)

for x in mydoc:
    print(x)


#$regex ====> finding the adresse tha begins with K
myquery = {"adressse":{"$regex":"^K"}}
mydoc = mycol.find(myquery)

for x in mydoc:
    print(x)

#sort() ===> Trier le resultat par order croissant
mydoc = mycol.find().sort("name")

for x in mydoc:
    print(x)


#trier par ordre decroissant
mydoc = mycol.find().sort("name",-1)

for x in mydoc:
    print(x)


#deleting the first occurance
myquery = { "adresse": "Highway 37"}
mycol.delete_one(myquery)

#deleting multiple/many documents

myquery = { "adressse": {"$regex":"^K"}}
x = mycol.delete_many(myquery)

#deleted_count ====> counts how many documents has been deleted
print(x.deleted_count,"documents Deleted")


#delete_many({}) ====> deleting all documents
#x = mycol.delete_many({})
#print(x.deleted_count,"Documents deleted")


#drop() =====> to delete the colluction
#x.drop()




#------------------updating------------------#
#$set ===> setting the new value
myquery = {"adressse":"Sousse"}
newvalues = {"$set":{"adressse":"Akouda"}}
mycol.update_one(myquery,newvalues)

#print customers after the update
for x in mycol.find():
    print(x)


#update_many()
#$regex ===> search for the adressses that begins with A and up
#$set ==> set the changes
myquery = {"adressse":{"$regex":"^A"}}
newvalues = {"$set":{"name":"ILEF"}}
x = mycol.update_many(myquery,newvalues)

print(x.modified_count,"Documents updated.")


#------------------limiting the results---------------#
#limit() ====> limit the results
myresult = mycol.find().limit(1)

#print the result
for x in myresult:
    print(x)



#pyrebase github

