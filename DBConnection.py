from pymongo import MongoClient
import gridfs

connection = MongoClient("localhost, 27017")
database = connection['Images']

fs = gridfs.GridFS(database)
file = "C:/Users/Willc/OneDrive Cegep de Sainte-Foy/CEGEP/Veille Technologique/ImageTest.jpg"

with open(file, 'rb') as f:
    content = f.read()

fs.put(content, filename="file")

