from flask import request, redirect, flash,session,g
from werkzeug.utils import secure_filename
import os, shutil, validators
from qa_over_docs import vector_db
from qa_over_docs import app, context, r_db, ALLOWED_EXTENSIONS, CONTEXT_FILE, SOURCES_FILE
from flask import jsonify

@app.route('/create_databases')
def create_collection():
    if not vector_db.collection_exists():
        vector_db.create_collections()
    context["collection_exists"] = True

    from qa_over_docs.relational_db import Question, Answer, Response
    r_db.create_all()

    flash("Databases successfully created", "success")
    return redirect("/")


def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/include_source", methods=['POST'])
def include_source():
    user_directory_path = session.get('user_directory_path', None)

    print(user_directory_path)
    if request.method == 'POST':
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            context["sources_to_add"].append(request.form["include-url"])
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(user_directory_path, filename)
            file.save(path)
            print(g.user_directory_path)
            context["sources_to_add"].append(filename)

            # Return a JSON response
            return jsonify({'status': 'success', 'message': 'File uploaded successfully', 'filename': filename})

    # Handle other cases or errors
    return jsonify({'status': 'error', 'message': 'Invalid request'})
@app.route("/clear_sources_to_add")
def clear_sources_to_add():
    context["sources_to_add"] = []
    shutil.rmtree(UPLOAD_FOLDER)
    os.mkdir(UPLOAD_FOLDER)
    return redirect("/")


@app.route("/add_sources", methods=['GET', 'POST'])
def add_sources():
    if request.method == 'POST':
        if context["sources_to_add"]:
            valid_sources = []
            
            for source in context["sources_to_add"]:
                if validators.url(source) or os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], source)):
                    valid_sources.append(source)
            if valid_sources:
                vector_db.add_sources(valid_sources)
                context["sources"].extend(valid_sources)
                clear_sources_to_add()
                flash("Successfully added sources", "success")
            else:
                flash("No valid sources provided", "warning")
        else:
            flash("No sources to add", "warning")
    return redirect("/")


@app.route("/remove_source/<int:index>")
def remove_source(index: int):
    source = context["sources"][index]
    vector_db.remove_source(source)
    flash(f"Successfully removed {source}", "primary")
    context["sources"].pop(index)
    return redirect("/")


@app.route("/delete_databases")
def delete_collection():
    vector_db.delete_collection()

    INSTANCE_DB = "instance/project.db"
    if os.path.exists(INSTANCE_DB):
        os.remove(INSTANCE_DB)

    if os.path.exists(CONTEXT_FILE):
        os.remove(CONTEXT_FILE)
    if os.path.exists(SOURCES_FILE):
        os.remove(SOURCES_FILE)

    context["collection_exists"] = False
    context["sources"] = []
    context["time_intervals"] = {}
    context["chat_items"] = []

    flash("Databases successfully deleted", "primary")
    return redirect("/")
