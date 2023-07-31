from flask import Flask, render_template
import pytesseract

from pasportocr import PasportOCR

app = Flask(__name__)


@app.route("/predict", methods=['GET'])
def predict():
    """For rendering results on HTML GUI"""
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'
    ocr = PasportOCR("static/Pasport3.png")
    pasport_data = ocr.exec()
    print(pasport_data)
    return render_template(
        'index.html', 
        surname=pasport_data["SRN"], 
        name=pasport_data["NME"], 
        second=pasport_data["SNM"], 
        year=pasport_data["BRD"], 
        series=pasport_data["SER"], 
        num=pasport_data["NUB"],
        gender = pasport_data["GND"],
        ctz=pasport_data["CTZ"],
        release=pasport_data["RLS"],
        code=pasport_data["COD"],
        fcs=pasport_data["FCS"],
    )


@app.route("/")
def hello_world():
    return render_template('index.html')
 

if __name__ == "__main__":
    app.run('localhost', 5000)