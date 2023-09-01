from flask import Flask
from flask_restful import Resource, Api



app = Flask(__name__)


api = Api(app)


productos = []


class Productos(Resource):
    def get(self, valor):
        for p in productos:
            if p == valor:
                return {'producto':p}
        return {'resultado':'Producto no encontrado'}
    
    
    def post(self, valor):
        productos = valor
         