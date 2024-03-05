from flask import Flask, jsonify

app = Flask(__name__)

stroke_age_data = [0.95011344, 0.9524488 , 0.95478417, 0.95711953, 0.9594549,
       0.96179026, 0.96412563, 0.96646099, 0.96879636, 0.97113172,
       0.97346709, 0.97580245, 0.97813782, 0.98047318, 0.98280855,
       0.98514391, 0.98747927, 0.98981464, 0.99215   , 0.99448537,
       0.99682073, 0.9991561 , 1.00149146, 1.00382683, 1.00616219,
       1.00849756, 1.01083292, 1.01316829, 1.01550365, 1.01783902,
       1.02017438, 1.02250975, 1.02484511, 1.02718047, 1.02951584,
       1.0318512 , 1.03418657, 1.03652193, 1.0388573 , 1.04119266,
       1.04352803, 1.04586339, 1.04819876, 1.05053412, 1.05286949,
       1.05520485, 1.05754022, 1.05987558, 1.06221095, 1.06454631,
       1.06688167, 1.06921704, 1.0715524 , 1.07388777, 1.07622313,
       1.0785585 , 1.08089386, 1.08322923, 1.08556459, 1.08789996,
       1.09023532, 1.09257069, 1.09490605, 1.09724142, 1.09957678,
       1.10191214, 1.10424751, 1.10658287, 1.10891824, 1.1112536 ,
       1.11358897, 1.11592433, 1.1182597 , 1.12059506, 1.12293043,
       1.12526579, 1.12760116, 1.12993652, 1.13227189, 1.13460725,
       1.13694262, 1.13927798, 1.14161334, 1.14394871, 1.14628407,
       1.14861944, 1.1509548 , 1.15329017, 1.15562553, 1.1579609 ,
       1.16029626, 1.16263163, 1.16496699, 1.16730236, 1.16963772,
       1.17197309, 1.17430845, 1.17664381, 1.17897918, 1.18131454]

@app.route('/project_API/project_api', methods=['GET'])
def get_data():
    return jsonify(stroke_age_data)

@app.route('/project_API/project_api/<int:id>', methods=['GET'])
def get_data_point(id):
    stroke_data_point = stroke_age_data[id]
    if stroke_data_point:
        return jsonify(stroke_data_point)
    return jsonify({"message":"keine daten für dieses alter"})

if __name__ == '__main__':
    app.run(debug=True)