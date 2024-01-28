from flask import Blueprint
from flask import request, send_from_directory, jsonify
from flask_socketio import SocketIO,emit
from app import socketio,executor
from utils.funcs import tc_legal, dir_convert
import threading

from SeaM_main.src.binary_class.run_calculate_flop import run_calculate_flop_bc
from SeaM_main.src.binary_class.run_model_reengineering import run_model_reengineering_bc

from GradSplitter_main.src.script.run_splitter import run_splitter_script
from GradSplitter_main.src.script.select_modules import run_select_modules_script
from GradSplitter_main.src.script.run_evaluate_modules import run_evaluate_modules_script
from GradSplitter_main.src.script.run_module_reuse_for_accurate_model import run_ensemble_modules_script

