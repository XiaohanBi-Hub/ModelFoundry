from flask import Blueprint
from flask import request, send_from_directory
from utils.funcs import tc_legal, dir_convert

download_bp = Blueprint('download_bp', __name__)

@download_bp.route('/download', methods=['POST'])
def download_file():
    data = request.get_json()
    print(data)
    model_file = data.get('modelFile')
    dataset_file = data.get('datasetFile')
    algorithm = data.get('algorithm')
    epoch = data.get('epoch')
    learning_rate = data.get('learningRate')
    direct_model_reuse = data.get('directModelReuse')
    target_class_str = data.get('targetClass')
    target_superclass_idx_str = data.get('targetSuperclassIdx')
    alpha = float(data.get('alpha')) if data.get('alpha') != '' else ''
    target_class = tc_legal(target_class_str)
    target_superclass_idx = tc_legal(target_superclass_idx_str)
    try:
        directory,filename = dir_convert(algorithm=algorithm, direct_model_reuse=direct_model_reuse, \
                                   model_file=model_file, dataset_file=dataset_file,target_class_str=target_class, \
                                   target_superclass_idx_str=target_class,lr_mask=learning_rate,alpha=alpha)

        print(f"Attempting to send from directory: {directory}, filename: {filename}")  # Debug line
        response = send_from_directory(directory, filename, as_attachment=True)
        response.headers["content-disposition"] = filename
        return response
    except Exception as e:
        print(f"An error occurred: {e}")  # Debug line
        return str(e), 400