from datetime import timedelta
from flask import Flask, request, render_template, send_from_directory,jsonify
from flask_cors import CORS
from flask_socketio import SocketIO,emit
from flask_executor import Executor
import os
# from SeaM_main.src.binary_class.run_calculate_flop import run_calculate_flop
from SeaM_main.src.multi_class.run_model_reengineering import run_model_reengineering_mc
from SeaM_main.src.multi_class.run_calculate_flop import run_calculate_flop_mc
from SeaM_main.src.binary_class.run_calculate_flop import run_calculate_flop_bc
from SeaM_main.src.binary_class.run_model_reengineering import run_model_reengineering_bc
# from SeaM_main.src.binary_class.run_calculate_time_cost import run_calculate_time_cost_bc
from SeaM_main.src.defect_inherit.run_reengineering_finetune import run_reengineering_finetune
from SeaM_main.src.defect_inherit.run_eval_robustness import run_eval_robustness
from SeaM_main.src.defect_inherit.run_standard_finetune import run_standard_finetune
from SeaM_main.src.binary_class.SeaM_reasoning import cifar10_inference
# Golbal config for SeaM
from SeaM_main.src.global_config import global_config as global_config_SeaM

from GradSplitter_main.src.script.run_train import run_train_script
from GradSplitter_main.src.script.run_splitter import run_splitter_script
from GradSplitter_main.src.script.select_modules import run_select_modules_script
from GradSplitter_main.src.script.run_evaluate_modules import run_evaluate_modules_script
from GradSplitter_main.src.script.run_module_reuse_for_accurate_model import run_ensemble_modules_script
from GradSplitter_main.src.script.run_module_reuse_for_new_task import run_reuse_modules_script_pair
# Golbal config for Grad
from GradSplitter_main.src.global_configure import global_config as global_config_Grad

from utils.funcs import tc_legal, dir_convert
from utils.download import download_bp
# from utils.benchmark import benchmark_bp
from utils.datasets_classes import datasets_classes


import threading

app = Flask(__name__)
app.config['EXECUTOR_TYPE'] = 'thread'
app.config['EXECUTOR_MAX_WORKERS'] = 5
CORS(app,expose_headers=['Content-Disposition'])
executor = Executor(app)
# Task dictionary for multi-tasks
tasks = {}


app.config['UPLOAD_FOLDER'] = 'uploads'
# create a SocketIO instance
socketio = SocketIO(app, cors_allowed_origins="*", allow_unsafe_werkzeug=True,\
                    expose_headers=['Content-Disposition'])
# Welcome page
@app.route('/')
def index():
    return "Welcome!"

# Connect socket
@socketio.on('connect')
def handle_connect():
    print("Client connected")
    socketio.emit('message', 'Successfully connected to the server!')

# Benchmark route
# app.register_blueprint(benchmark_bp)

# The main running route
@app.route('/run_model', methods=['POST'])
def run_model():
    data = request.get_json()
    print(data)
    model_file = data.get('modelFile')
    dataset_file = data.get('datasetFile')
    algorithm = data.get('algorithm')
    epoch = float(tc_legal(data.get("epoch")))
    learning_rate = float(data.get('learningRate'))
    direct_model_reuse = data.get('directModelReuse')
    target_class_str = data.get('targetClass')
    target_superclass_idx_str = data.get('targetSuperclassIdx')
    alpha = float(data.get('alpha')) if data.get('alpha') != '' else ''
    target_class = tc_legal(target_class_str)
    target_superclass_idx = tc_legal(target_superclass_idx_str)
    socketio.emit('get_progress_percentage', 100)
    if algorithm=='SEAM':
        try:
            # Callback function for model results
            def callback(**kwargs):
                messages = {
                    'm_total_flop_dense': 'FLOPs Dense: {:.2f}M',
                    'm_total_flop_sparse': 'FLOPs Sparse: {:.2f}M',
                    'perc_sparse_dense': 'FLOPs % (Sparse / Dense): {:.2%}',
                    'acc_pre': 'Pretrained Model ACC: {:.2%}',
                    'acc_reeng': 'Module ACC: {:.2%}',
                    'sum_masks':'Weights of original model:{:.2f}million',
                    'sum_mask_ones':'Weights of module:{:.2f}million',
                    'weight_retain_rate':"Weight retain rate:{:.2%}",
                    # For Defect Inheritance reengineering
                    'best_epoch_step1': 'Best epoch in step 1:{}',
                    'best_acc_step1': 'Best acc in step 1:{:.2%}',
                    'best_epoch_step3': 'Best epoch in step 3:{}',
                    'best_acc_step3': 'Best acc in step 3:{:.2%}',
                    # For evaluate robustness
                    'clean_top1' : 'Clean Top-1: {:.2f}',
                    'adv_sr': 'Attack Success Rate: {:.2f}',
                    'step_1': 'Step 1:Finetuning the output layer......',
                    'step_2': 'Step 2:Decomposing the fine-tuned model obtained in Step 1......',
                    'step_3': 'Step 3:Finetune according to decomposing results......',
                }
                for key, message in messages.items():
                    if key in kwargs:
                        socketio.emit('model_result', message.format(kwargs[key]))
            def get_epochs(epoch,n_epoch=30):
                epoch_percentage = (epoch/n_epoch)*100
                socketio.emit('get_progress_percentage', epoch_percentage)
                print(f"Epoch percentage:{epoch_percentage:.2f}%")
                return epoch_percentage
            def run():
                if direct_model_reuse=='Binary Classification':
                    socketio.emit('message','\n Decomposing Model, Please Wait!!!')
                    # run_model_reengineering_bc(model=model_file, dataset=dataset_file, 
                    #                 target_class=target_class,lr_mask=learning_rate, alpha=alpha, 
                    #                 n_epochs=300,get_epochs=get_epochs)
                    socketio.emit('message','\n Model is ready, waiting for calculating flops......')
                    run_calculate_flop_bc(model=model_file, dataset=dataset_file, 
                                target_class=target_class, lr_mask=learning_rate, alpha=alpha,
                                callback=callback)
                    print("Calculating flops done!")
                    socketio.emit('message','\n Calculating flops done!')
                    
                elif direct_model_reuse=='Multi-Class Classification':
                    socketio.emit('message','\n Decomposing Model, Please Wait!!!')  
                    run_model_reengineering_mc(model=model_file, dataset=dataset_file, 
                                target_superclass_idx=target_superclass_idx,
                                lr_mask=learning_rate, alpha=alpha, get_epochs=get_epochs)
                    socketio.emit('message','\n Model is ready, waiting for calculating flops......')
                    run_calculate_flop_mc(model=model_file, dataset=dataset_file, 
                                target_superclass_idx=target_superclass_idx, 
                                lr_mask=learning_rate, alpha=alpha,callback=callback)
                elif direct_model_reuse == 'Defect Inheritance':
                    # 1.Re-engineer ResNet18-ImageNet and then fine-tune 
                    # the re-engineered model on the target dataset Scenes.
                    # 
                    # 2. Compute the defect inheritance rate of fine-tuned 
                    # re-engineered ResNet18-Scenes.
                    # 
                    # 3. Fine-tune the original ResNet18-ImageNet on the 
                    # target dataset Scenes.
                    # 
                    # 4. Compute the defect inheritance rate of fine-tuned 
                    # original ResNet18-Scenes.
                    socketio.emit('message','\n ## Decomposing Model...... ##')
                    socketio.emit('message','\n ## Process might be stopped before 100% when it fits the target! ##')
                    run_reengineering_finetune(model=model_file, dataset=dataset_file,
                           lr_mask=0.05, alpha=0.5, prune_threshold=0.6, callback=callback,get_epochs=get_epochs)
                    socketio.emit('message','\n ## Evaluating robustness...... ##')
                    run_eval_robustness(model=model_file, dataset=dataset_file, 
                            eval_method="seam", lr_mask=0.05, alpha=0.5, prune_threshold=0.6,callback=callback)
                    socketio.emit('message','\n ## Start Fine-tuning. ##')
                    run_standard_finetune(model=model_file, dataset=dataset_file)
                    socketio.emit('message','\n ## Finish Fine-tuning. ##')
                    socketio.emit('message','\n ## Evaluating robustness...... ##')
                    run_eval_robustness(model=model_file, dataset=dataset_file, eval_method="standard")
                    socketio.emit('message','\n ## Evaluate Done! ##')
                else:
                    print("Model reuse type error!!")
                    return ValueError
            future = executor.submit(run)
            tasks[str(id(future))] = future
            print(tasks)
            return {'logs': "Model run successfully", 'isModelReady': True, 'task_id': future.id}, 200
        except Exception as e:
            return {'logs': str(e), 'isModelReady': False}, 500
    elif algorithm=='GradSplitter':
        try:
            def callback(**kwargs):
                messages = {
                    'best_modules':'Best Module:{}',
                    'best_epoch':'Best Epoch:{}',
                    'best_acc':'Best Acc:{:.2%}',
                    'best_avg_kernel':'Best Averag Kernel:{:.2f}',
                }
                # socketio.emit('model_result', f'Best Module: {best_modules}')
                # socketio.emit('model_result', f'Best Epoch: {best_epoch}')
                # socketio.emit('model_result', f'Best Acc: {best_acc * 100:.2f}%')
                # socketio.emit('model_result', f'Best Averag Kernel: {best_avg_kernel:.2f}')
                for key, message in messages.items():
                    if key in kwargs:
                        socketio.emit('message', message.format(kwargs[key]))
            def get_epochs(epoch,n_epoch=145):
                epoch_percentage = (epoch/n_epoch)*100
                socketio.emit('get_progress_percentage', epoch_percentage)
                print(f"Epoch percentage:{epoch_percentage:.2f}%")
                return epoch_percentage
            def run():
                socketio.emit('message','\n Decomposing Model, Please Wait!!!')
                # run_splitter_script(model=model_file,dataset=dataset_file,callback=callback, get_epochs=get_epochs)
                socketio.emit('message','\n Decomposing Done!')
                socketio.emit('message','\n Selecting Modules, Please Wait!!!')
                run_select_modules_script(model=model_file,dataset=dataset_file)
                socketio.emit('message','\n Modules Selected!!!')
            future = executor.submit(run)
            tasks[str(id(future))] = future
            return {'logs': "Model run successfully", 'isModelReady': True, 'task_id': future.id}, 200
        except Exception as e:
            return {'logs': str(e), 'isModelReady': False}, 500
        return

# Search module route
@app.route('/search_module', methods=['POST'])
def search_module():
    data = request.get_json()
    search_class = data.get('searchClass')

    # class_tag = "classes" if target_class>=0 else "superclasses"
    # classes_list = eval(f'datasets_classes.get_classes().{dataset_file}_{class_tag}')

    # 用于存放在class与superclass搜索到的内容，形式为{"cat":"cifar10_classes"}
    classes_search_result = {}
    superclasses_search_result = {}
    result_list = []
    datasets_classes.get_classes()
    for name in ['cifar10_classes','cifar100_classes','svhn_classes']:
        for idx, class_label in enumerate(eval(f"datasets_classes.{name}")):
            if search_class in class_label:
                classes_search_result[class_label] = [idx,name]
    for idx, class_label in enumerate(datasets_classes.cifar100_superclasses):
        if search_class in class_label:
            superclasses_search_result[class_label] = [idx, "cifar100_superclasses"]
    # 对于imagenet，查超类，如果超类没有就查对应子类，子类或超类有一个就返回超类的label
    for idx, class_label in enumerate(datasets_classes.imagenet_superclasses):
        if search_class in class_label:
            superclasses_search_result[class_label] = [idx, "imagenet_superclasses"]
        else:
            for item in datasets_classes.imagenet_superclasses_classes[class_label]:
                if search_class in item:
                    superclasses_search_result[class_label] = [idx, "imagenet_superclasses_subclass"]

    for key, value in classes_search_result.items():
        result_dic = {}
        result_dic['module_name'] = key
        result_dic['dataset'] = value[1]
        result_dic['idx'] = value[0]
        result_dic['tag'] = "classes"
        result_list.append(result_dic)
    for key, value in superclasses_search_result.items():
        result_dic = {}
        result_dic['module_name'] = key
        result_dic['dataset'] = value[1]
        result_dic['idx'] = value[0]
        result_dic['tag'] = "superclasses"
        result_list.append(result_dic)
    #{'module_name': 'cat'
    #  'dataset' : 'cifar10_classes'
    #   'tag' : 'classes'/'superclasses'
    #   }  
    return jsonify(result_list)

# Reuse module
@app.route('/run_reuse', methods=['POST'])
def run_reuse():
    data = request.get_json()
    print(data)
    model_file = data.get('modelFile')
    dataset_file = data.get('datasetFile')
    algorithm = data.get('algorithm')
    epoch = data.get('epoch')
    reuseMethod = data.get('reuseMethod')
    cifarclass = data.get('cifarclass')
    svhnclass = data.get('svhnclass')
    if reuseMethod == "More Accurate":
        try:
            def callback(**kwargs):
                # socketio.emit('reuse_result', f'Best acc: {acc * 100:.2f}%')
                messages = {
                    'acc':'Best Acc:{:.2%}',
                    'module_acc':'{}',
                    'model_acc':'{}',
                    'avg_model_acc':'Average pretrained model Acc:{:.2%}'
                }
                for key, message in messages.items():
                    if key in kwargs:
                        socketio.emit('message', message.format(kwargs[key]))
            def run():
                socketio.emit('message','\n Evaluating Modules......')
                run_evaluate_modules_script(model=model_file,dataset=dataset_file,callback=callback)
                socketio.emit('message','\n Trying to ensemble a more accurate model......')
                run_ensemble_modules_script(model=model_file,dataset=dataset_file,callback=callback)
                socketio.emit('message','\n New accuate model ensembled!!!')
            threading.Thread(target=run).start()
            return {'logs': "Model run successfully", 'isModelReady': True}, 200
        except Exception as e:
            return {'logs': str(e), 'isModelReady': False}, 500
    elif reuseMethod == "For New Task":
        print("Model reuse for new task......")
        try:
            def callback(acc):
                socketio.emit('reuse_result', f'ACC: {acc:.2f}')
            def run():
                socketio.emit('reuse_message',f'\n Evaluating and selecting modules......')
                run_reuse_modules_script_pair(model_file,cifarclass,svhnclass,callback=callback)
                socketio.emit('reuse_message',f'\n Select module {cifarclass} for CIFAR, {svhnclass} for SVHN!')
            threading.Thread(target=run).start()
            return {'reuse_message': "Model run successfully", 'isModelReady': True}, 200
        except Exception as e:
            return {'reuse_message': str(e), 'isModelReady': False}, 500
    return

# Download module
app.register_blueprint(download_bp)

@app.route('/run_deployment', methods=['POST'])
def run_deployment():
    data = request.get_json()
    img_name = data.get('image')
    class_name,class_num = cifar10_inference.predict(f'image/{img_name}.png')
    print(class_name)
    result = class_name if class_num!=0 else f"Not cat"
    socketio.emit('deployment_result',f'{result}')
    return class_name

@app.route('/list_tasks', methods=['GET'])
def list_tasks():
    task_statuses = {}
    for task_id in tasks.keys():
        if tasks[task_id].running():
            status = 'running'
        elif tasks[task_id].done():
            status = 'done'
        else:
            status = 'pending'
        task_statuses[task_id] = status
    return jsonify(task_statuses)

if __name__ == '__main__':
    socketio.run(app, debug=True)
