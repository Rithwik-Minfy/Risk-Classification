import os
import joblib
import pandas as pd
from flask import Flask, request, render_template, redirect, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'replace-with-a-secure-random-key'

# Upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- LOAD TRANSFORMERS & SELECTOR ---
pt       = joblib.load('saved_models/pt.pkl')
rs       = joblib.load('saved_models/rs.pkl')
ss       = joblib.load('saved_models/ss.pkl')
selector = joblib.load('saved_models/selector.pkl')

# --- LOAD YOUR TRAINED MODELS ---
models = {}
for fn in os.listdir('saved_models'):
    if fn.endswith('_model.pkl'):
        name = fn.replace('_model.pkl','')
        models[name] = joblib.load(os.path.join('saved_models', fn))


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def preprocess_and_select(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Drop columns unseen during training
    df = df.drop(['ID', 'ZIP Code', 'Personal Loan'], axis=1, errors='ignore')

    # 2) Apply same transforms
    rb_cols  = ['CCAvg', 'Mortgage']
    std_cols = ['Income', 'Experience', 'Age']

    df[rb_cols]  = pt.transform(df[rb_cols])
    df[rb_cols]  = rs.transform(df[rb_cols])
    df[std_cols] = ss.transform(df[std_cols])

    # 3) RFE feature‐selection
    X_sel = selector.transform(df)
    cols  = [f'F{i}' for i in range(X_sel.shape[1])]
    return pd.DataFrame(X_sel, columns=cols)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Read and preprocess
            df = pd.read_csv(filepath)
            X  = preprocess_and_select(df.copy())

            # Predict with each model
            for name, model in models.items():
                df[f'Pred_{name}'] = model.predict(X)

            # Render results
            return render_template(
                'results.html',
                table=df.to_html(classes='table table-striped', index=False),
                models=list(models.keys())
            )

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
