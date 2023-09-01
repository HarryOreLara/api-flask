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
        producto = valor
        productos.append(producto)
        return {'resultado':'Producto agregado correctamente'}
    

    def delete(self, valor):
        for indice, p, in enumerate(productos):
            if p == valor:
                productos.pop(indice)
                return {'resultado':'Producto eliminado correctamente'}


class Listar(Resource):
    def get(self):
        return {'resultado':productos}
    
api.add_resource(Productos, '/producto/<string:valor>')
api.add_resource(Listar, '/listar')



if __name__ == '__main__':
    app.run(debug=True)