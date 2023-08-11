import os
import io
import sys
import time
import pathlib
from zipfile import ZipFile
from flask import Flask, request, render_template, send_file, abort
from functools import wraps

par_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(par_dir)

TEMPLATE_PATH = os.path.join(par_dir, 'templates/')
print(TEMPLATE_PATH)
app = Flask(__name__, template_folder=TEMPLATE_PATH)


# The actual decorator function
def require_appkey(view_function):
    @wraps(view_function)
    # the new, post-decoration function. Note *args and **kwargs here.
    def decorated_function(*args, **kwargs):
        print(request.args.get('key'))
        if request.args.get('key') and request.args.get('key') == os.environ['KEY']:
            print(request.args.get('key'))
            return view_function(*args, **kwargs)
        else:
            abort(401)

    return decorated_function


@app.route('/upload')
@require_appkey
def upload():
    import os
    return render_template('upload.html', key=os.environ['KEY'], port=os.environ['EXT_SERVICE_PORT'])


@app.route('/uploader', methods=['GET', 'POST'])
@require_appkey
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(par_dir + '/' + f.filename)
        return 'File uploaded successfully'


@app.route("/run_foo", methods=["GET"])
@require_appkey
def run_foo():
    return None


@app.route("/results", methods=["GET"])
@require_appkey
def get_results_file():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = "results_{}.zip".format(timestr)
    file_path = pathlib.Path(par_dir + '/results')
    memory_file = io.BytesIO()

    with ZipFile(memory_file, mode='w') as z:
        for f_name in file_path.iterdir():
            print(f_name)
            z.write(f_name)

    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        attachment_filename=filename
    )


if __name__ == '__main__':
    app.run(debug=True)
