from flask import Flask, render_template, request

from werkzeug.utils import secure_filename
import os
from PIL import Image, ImageOps
from keras.models import load_model

app = Flask(__name__, static_folder = 'files')
app.config['UPLOAD_FOLDER'] = 'files'

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/result', methods = ['POST'])
def result():
	# Saving User Upload Image
	image_file = request.files['file']
	save_location = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
	image_file.save(save_location)

	# Performing Image Pre-Processing
	image = Image.open(image_file)
	image = image.resize((28, 28)) # Resizing Image
	image = image.convert('L') # Grey Scaling Image
	image = ImageOps.invert(image) # Invert Grey Scale
	img = np.array(image)

	# Performing Prediction
	model = load_model('model.h5')
	res = model.predict(img.reshape(-1, 28, 28, 1))
	predicted_result = res.argmax()

	return render_template('result.html', pred=predicted_result, file=image_file.filename)


if __name__ == '__main__':
   app.run(debug = True)