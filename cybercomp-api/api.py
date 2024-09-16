from pathlib import Path
from flask import Flask, jsonify
import yaml

app = Flask(__name__)


# Helper function to read YAML file
def read_yaml_file(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


# Route to serve first YAML file as JSON
@app.route("/models", methods=["GET"])
def get_file1():
    data = read_yaml_file("file1.yml")
    return jsonify(data)


# Route to serve second YAML file as JSON
@app.route("/engines", methods=["GET"])
def get_file2():
    data = read_yaml_file("file2.yml")
    return jsonify(data)


# Route to serve third YAML file as JSON
@app.route("/sources", methods=["GET"])
def get_file3():
    data = read_yaml_file("file3.yml")
    return jsonify(data)


if __name__ == "__main__":

    lib_dir = Path(__file__).parent
    data_dir = lib_dir / "database"
    types_dir = lib_dir / "generated" / "types"
    model_dir = lib_dir / "generated" / "models"
    engine_dir = lib_dir / "generated" / "engines"
    types_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_dir.mkdir(parents=True, exist_ok=True)

    app.run(debug=True)
