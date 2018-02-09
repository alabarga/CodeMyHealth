# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, abort
import flask_restful
import classifier
import json
import pdb


app = Flask(__name__)
api = flask_restful.Api(app)
clasificador = None


@app.route("/predict", methods=["POST"])
def predict():
    if not request.json:
        abort(400)
    content = request.get_json()
    if content['texto']:
        texto = content['texto'].encode('utf-8')
        print texto
        salida = clasificador.explotacion(texto)
        salida = json.dumps(salida, ensure_ascii=False)
        return salida
    else:
        return 'ERROR'


if __name__ == "__main__":
    global clasificador
    clasificador = classifier.Classifier()
    app.run('0.0.0.0', port=5000, debug=True)

