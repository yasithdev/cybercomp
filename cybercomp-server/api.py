from flask import Flask, jsonify

from dataloader import DataLoader

app = Flask(__name__)
dl = DataLoader()


@app.route("/types", methods=["GET"])
def get_all_types():
    data = dl.get_all_types()
    return jsonify({k: v.model_dump() for k, v in data.items()})


@app.route("/models", methods=["GET"])
def get_all_models():
    data = dl.get_all_models()
    return jsonify({k: v.model_dump() for k, v in data.items()})


@app.route("/engines", methods=["GET"])
def get_all_engines():
    data = dl.get_all_engines()
    return jsonify({k: v.model_dump() for k, v in data.items()})


@app.route("/sources", methods=["GET"])
def list_all_sources():
    data = dl.list_all_sources()
    return jsonify({k: v for k, v in data.items()})


@app.route("/status", methods=["GET"])
def health_check():
    return jsonify(dict(status=True))


if __name__ == "__main__":
    app.run(port=8765, debug=True)
