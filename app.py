from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model using joblib
model = joblib.load('models/trained_model_1.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve input values from the form
            area = float(request.form['area'])
            bedrooms = request.form['bedrooms']
            bathrooms = request.form['bathrooms']
            stories = request.form['stories']
            mainroad = request.form['mainroad']
            guestroom = request.form['guestroom']
            basement = request.form['basement']
            hotwaterheating = request.form['hotwaterheating']
            airconditioning = request.form['airconditioning']
            parking = request.form['parking']
            prearea = request.form['prearea']
            furnishingstatus = request.form['furnishingstatus']
            price_per_sqrt = float(request.form['price_per_sqrt'])

            # Convert "Yes" to 1 and "No" to 0
            bedrooms = 1 if bedrooms == "Yes" else 0
            bathrooms = 1 if bathrooms == "Yes" else 0
            stories = 1 if stories == "Yes" else 0
            mainroad = 1 if mainroad == "Yes" else 0
            guestroom = 1 if guestroom == "Yes" else 0
            basement = 1 if basement == "Yes" else 0
            hotwaterheating = 1 if hotwaterheating == "Yes" else 0
            airconditioning = 1 if airconditioning == "Yes" else 0
            parking = 1 if parking == "Yes" else 0
            prearea = 1 if prearea == "Yes" else 0
            furnishingstatus = {'Furnished': 2, 'Semi Furnished': 1}.get(furnishingstatus, 0)


            print(area,bedrooms, bathrooms,  stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, prearea, furnishingstatus, price_per_sqrt)

            # Make a prediction using the loaded model
            input_data = pd.DataFrame({
                'area': [area],
                'bedrooms': [bedrooms],
                'bathrooms': [bathrooms],
                'stories': [stories],
                'mainroad': [mainroad],
                'guestroom': [guestroom],
                'basement': [basement],
                'hotwaterheating': [hotwaterheating],
                'airconditioning': [airconditioning],
                'parking': [parking],
                'prearea': [prearea],
                'furnishingstatus': [furnishingstatus],
                'price_per_sqrt': [price_per_sqrt]
            })

            # Convert categorical variables to binary (0/1)
            input_data = pd.get_dummies(input_data, drop_first=True)

            predicted_price = model.predict(input_data)[0]

            return render_template('index.html', prediction=f'Predicted Price: ${predicted_price:.2f}')
        except Exception as e:
            return render_template('index.html', prediction=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)
