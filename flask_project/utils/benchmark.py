from flask import Blueprint
from flask import request, send_from_directory, jsonify
from flask_socketio import SocketIO,emit
from app import socketio, executor
from utils.funcs import tc_legal, dir_convert
import threading

from SeaM_main.src.binary_class.run_calculate_flop import run_calculate_flop_bc
from SeaM_main.src.binary_class.run_model_reengineering import run_model_reengineering_bc

from GradSplitter_main.src.script.run_splitter import run_splitter_script
from GradSplitter_main.src.script.select_modules import run_select_modules_script
from GradSplitter_main.src.script.run_evaluate_modules import run_evaluate_modules_script
from GradSplitter_main.src.script.run_module_reuse_for_accurate_model import run_ensemble_modules_script


benchmark_bp = Blueprint('run_model_bp', __name__)

# The benchmark route
@benchmark_bp.route('/benchmark', methods=['POST'])
def benchmark():
    data = request.get_json()
    print(data)
    model_file = data.get('modelFile')
    dataset_file = data.get('datasetFile')
    algorithm = data.get('algorithm')
    learning_rate = float(data.get('learningRate'))
    direct_model_reuse = data.get('directModelReuse')
    target_class = 0
    alpha = float(data.get('alpha')) if data.get('alpha') != '' else ''
    if algorithm=='SEAM':
        try:
            def callback(m_total_flop_dense, m_total_flop_sparse, 
                        perc_sparse_dense, acc_reeng, acc_pre):
                socketio.emit('seam_result', f'FLOPs Dense: {m_total_flop_dense:.2f}M')
                socketio.emit('seam_result', f'FLOPs Sparse: {m_total_flop_sparse:.2f}M')
                socketio.emit('seam_result', f'FLOPs % (Sparse / Dense): {perc_sparse_dense:.2%}')
                socketio.emit('seam_result', f'Pretrained Model ACC: {acc_pre:.2%}')
                socketio.emit('seam_result', f'Module ACC: {acc_reeng:.2%}')
            def run():
                try:
                    if direct_model_reuse=='Binary Classification':
                        socketio.emit('seam_message','\nDecomposing Model, Please Wait!!!')
                        print("\nDecomposing Model, Please Wait!!!")
                        run_model_reengineering_bc(model=model_file, dataset=dataset_file, 
                                        target_class=target_class,lr_mask=learning_rate, alpha=alpha)
                        socketio.emit('seam_message','\nModel is ready, waiting for calculating flops......')
                        run_calculate_flop_bc(model=model_file, dataset=dataset_file, 
                                    target_class=target_class, lr_mask=learning_rate, alpha=alpha,
                                    callback=callback)
                    else:
                        print("Model reuse type error!!")
                        return ValueError
                except Exception as e:
                    print(f"Exception in run function: {e}")
                
            # start a new thread to run the model
            print("About to start the run thread.")  # Debug line
            threading.Thread(target=run).start()
            print("Run thread started.")  # Debug line
            return {'logs': "Model run successfully", 'isModelReady': True}, 200
        except Exception as e:
            return {'logs': str(e), 'isModelReady': False}, 500
    elif algorithm=='GradSplitter':
        try:
            def callback(best_modules,best_epoch,best_acc,best_avg_kernel):
                socketio.emit('grad_result', f'Best Module: {best_modules}')
                socketio.emit('grad_result', f'Best Epoch: {best_epoch}')
                socketio.emit('grad_result', f'Best Acc: {best_acc * 100:.2f}%')
                socketio.emit('grad_result', f'Best_avg_kernel: {best_avg_kernel:.2f}')
            def run():
                socketio.emit('grad_message','\n Decomposing Model, Please Wait!!!')
                run_splitter_script(model=model_file,dataset=dataset_file)
                socketio.emit('grad_message','\n Decomposing Done!')
                socketio.emit('grad_message','\n Selecting Modules, Please Wait!!!')
                run_select_modules_script(model=model_file,dataset=dataset_file)
                socketio.emit('grad_message','\n Modules Selected!!!')
                socketio.emit('grad_message','\n Evaluating Modules......')
                run_evaluate_modules_script(model=model_file,dataset=dataset_file)
                socketio.emit('grad_message','\n Trying to ensemble a more accurate model......')
                run_ensemble_modules_script(model=model_file,dataset=dataset_file)
                socketio.emit('grad_message','\n New accuate model ensembled!!!')

            # start a new thread to run the model
            threading.Thread(target=run).start()
            return {'logs': "Model run successfully", 'isModelReady': True}, 200
        except Exception as e:
            return {'logs': str(e), 'isModelReady': False}, 500
    return jsonify({'message': 'Benchmark completed'})