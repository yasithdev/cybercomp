from flask import Flask, jsonify

from dataloader import DataLoader

app = Flask(__name__)
dl = DataLoader()


@app.route("/types", methods=["GET"])
def get_all_types():
    data = dl.get_all_types()
    return jsonify(data)


@app.route("/models", methods=["GET"])
def get_all_models():
    data = dl.get_all_models()
    return jsonify(data)


@app.route("/engines", methods=["GET"])
def get_all_engines():
    data = dl.get_all_engines()
    return jsonify(data)


@app.route("/sources", methods=["GET"])
def list_all_sources():
    data = dl.list_all_sources()
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
