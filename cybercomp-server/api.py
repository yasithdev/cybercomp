import io
import zipfile
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for, send_from_directory

from dataloader import DataLoader

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit
app.secret_key = "supersecretkey"


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


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/browse/<dir_path>", methods=["GET"])
def browse(dir_path: str):
    base_path = Path(app.config["UPLOAD_FOLDER"])
    abs_dir_path = base_path / dir_path
    files = list(abs_dir_path.glob("**/*"))
    filenames = map(lambda x: x.relative_to(base_path).as_posix(), files)
    return render_template("browse.html", files=filenames)


@app.route("/download/<path:file_path>", methods=["GET"])
def download(file_path: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], file_path, as_attachment=True)


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        flash("No file part")
        return redirect("/")

    file = request.files["file"]
    base_path = Path(app.config["UPLOAD_FOLDER"])
    rel_path = Path(file.filename or "")

    if rel_path.stem == "":
        flash("No selected file")
        return redirect("/")

    if rel_path.suffix == ".zip":
        # Use BytesIO to handle file in memory
        try:
            with zipfile.ZipFile(io.BytesIO(file.read())) as zip_ref:
                # Extract all files to the specified folder
                zip_ref.extractall(base_path / rel_path.stem)
            flash("File successfully uploaded and extracted")
        except zipfile.BadZipFile:
            flash("Invalid zip file")

        return redirect(f"/browse/{rel_path.stem}")
    else:
        flash("Invalid file format, only ZIP files are allowed")
        return redirect("/")


if __name__ == "__main__":
    app.run(port=8765, debug=True)
