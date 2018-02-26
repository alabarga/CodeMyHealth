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
    print content
    if content['texto']:
        texto = content['texto'].encode('utf-8')
        lang = content['lang']
        print texto
        return json.dumps(clasificador.explotacion_conjunta(texto, lang))
    else:
        abort(400)


if __name__ == "__main__":
    global clasificador
    clasificador = classifier.Classifier()
    app.run('0.0.0.0', port=5000, debug=True)

