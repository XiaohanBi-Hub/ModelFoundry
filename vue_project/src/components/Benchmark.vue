<template>
    <el-container>
        <!-- Header -->
        <el-header style="height: 85px;">
        <span style="margin-top: 0;margin-left: 5%;" @click="jumptohome"> 
            ModelFoundry</span> </el-header>

        <!-- Page Body-->
        <el-main>
            <h2 style="text-align: left;margin-left: 10.5%;margin-top: 2%;margin-bottom: 3%;">
                <i class="el-icon-arrow-left" style="margin-right: 5px;" @click="jumptohome"></i>
                Run Benchmark</h2>

            <!-- Main Body -->
            <div class="main-body" style="margin-left: 10%;margin-right: 10%;margin-top: 50px;">
                <el-divider style="width: 80%;margin-left: 10%;"></el-divider>
                <el-row>
                    <!-- SEAM -->
                    <el-col :span="12"><div>
                        <h3 style="margin-left: 10%;margin-right: 10%;margin-top: 20px;float:left;width:100px">
                            SeaM</h3>
                        <el-button style="float:right;margin-top: 10px;margin-right: 10%;width: 200px;" type="success" 
                        @click="SEAMdialogVisible = true"> RUN Benchmark </el-button>

                        <div class="tableBody" style="width: 95%;margin-left: 5%;" >
                            <el-table :data="tableDataSEAM" stripe style="width: 100%;" :row-class-name="coloredrow"
                            :span-method="objectSpanMethod" :header-cell-style="{textAlign: 'center', height: '100px'}" 
                            :cell-style="{'text-align':'center'}">
                                <el-table-column :render-header="renderheader" 
                                    prop="targetProblem" :label="'Target // Problem'" width="175">
                                </el-table-column>
                                <el-table-column :render-header="renderheader" 
                                    prop="modelName" :label="'Model // Name'" width="160">
                                </el-table-column>

                                <el-table-column :render-header="renderheader"
                                    prop="targetClass" :label="'Default Target Class'">
                                </el-table-column>
                                <el-table-column :render-header="renderheader" 
                                    prop="learningRate" :label="'Learning // Rate'" >
                                </el-table-column>
                                <el-table-column :render-header="renderheader" 
                                    prop="alpha" :label="'Alpha'" >
                                </el-table-column>
                            </el-table>

                        </div>
                    </div></el-col>

                    <!-- GRAD-->
                    <el-col :span="12"><div >
                        <h3 style="margin-left: 10%;margin-right: 10%;margin-top: 20px;float:left;width:100px">
                            GradSplitter</h3>
                        <el-button style="float:right;margin-top: 10px;margin-right: 10%;width: 200px;" type="success"
                        @click="GRADdialogVisible = true"> RUN Benchmark </el-button>
                        
                        <div class="tableBody" style="width: 95%;margin-left: 5%;" >
                            <el-table :data="tableDataGrad" stripe style="width: 100%"  :row-class-name="coloredrow"
                            :header-cell-style="{textAlign: 'center', height: '100px'}" 
                            :cell-style="{'text-align':'center'}">
                                <el-table-column 
                                    prop="modelName" label="Model Name" width="150">
                                </el-table-column>
                                <el-table-column :render-header="renderheader" 
                                    prop="learningRate_head" :label="'Learning Rate // (head)'">
                                </el-table-column>
                                <el-table-column  :render-header="renderheader" width="90"
                                    prop="learningRate_modularity" :label="'Learning Rate // (modular)'">
                                </el-table-column>
                                <el-table-column :render-header="renderheader"
                                    prop="epochs_head" :label="'Epochs // (head)'">
                                </el-table-column>
                                <el-table-column :render-header="renderheader"
                                    prop="epochs_modularity" :label="'Epochs // (modular)'">
                                </el-table-column>
                            </el-table>

                        </div>
                    </div></el-col>
                </el-row>


                <!-- BENCHMARK(SEAM) -->
                <el-dialog
                    title="Run SeaM Benchmark"
                    :visible.sync="SEAMdialogVisible"
                    width="30%">
                    <span>
                        <el-descriptions title="Benchmark Information" border size="small">
                            <!-- Algorithm, model and dataset-->
                            <el-descriptions-item label="Algortihm"  span="2">
                                <el-tag effect="dark" :hit="true" :color="seambenchmarkdata.algorithm === 'SEAM' ?'#00bcd4':'#ffc107'"  
                                style="font-weight: bolder;" >
                                    {{ seambenchmarkdata.algorithm }}</el-tag></el-descriptions-item>

                            <!-- Target Class(SEAM) -->
                            <el-descriptions-item v-if="seambenchmarkdata.algorithm === 'SEAM'" span="2"
                                label="Model Reuse Method">{{ seambenchmarkdata.directModelReuse }}</el-descriptions-item>

                            <el-descriptions-item label="Model" span="2">{{seambenchmarkdata.modelFile}}</el-descriptions-item>
                            <el-descriptions-item label="Dataset" span="2">{{ seambenchmarkdata.datasetFile }}</el-descriptions-item>
                        
                            <!-- Epoch(Grad) -->
                            <el-descriptions-item v-if="seambenchmarkdata.algorithm === 'GradSplitter'" 
                                label="Epochs">{{ seambenchmarkdata.epoch }}</el-descriptions-item>
                            
                            <el-descriptions-item v-if="seambenchmarkdata.algorithm === 'SEAM' && seambenchmarkdata.directModelReuse === 'Multi-Class Classification'"
                            span="4" label="Default Target Superclass Idx" >{{ seambenchmarkdata.targetSuperclassIdx }}</el-descriptions-item>
                            <el-descriptions-item v-if="seambenchmarkdata.algorithm === 'SEAM' && seambenchmarkdata.directModelReuse === 'Binary Classification'"
                            span="4" label="Default Target Class Idx">{{ seambenchmarkdata.targetClass }}</el-descriptions-item>
                            
                            <!-- learningRate(Both) -->
                            <el-descriptions-item label="Learning rate"  span="2">{{ seambenchmarkdata.learningRate }}</el-descriptions-item>
                            
                            <!-- alpha(SEAM) -->
                            <el-descriptions-item v-if="seambenchmarkdata.algorithm === 'SEAM'" 
                                label="Alpha"  span="2">{{ seambenchmarkdata.alpha }}</el-descriptions-item>

                        </el-descriptions>
                    </span>
                    <span> <el-input readonly resize="none"
                        type="textarea"
                        rows="7"
                        v-model="seambenchmarklogs"
                        style="width: 100%;margin-top: 40px;">
                    </el-input></span>
                    <span slot="footer" class="dialog-footer">
                        <el-button type="primary" @click="openSEAMdialog">RUN</el-button>
                        <el-button type="warning" @click="SEAMdialogVisible = false">Close</el-button>
                    </span>
                </el-dialog>

                <!-- BENCHMARK(Grad)-->
                <el-dialog
                    title="Run GradSplitter Benchmark"
                    :visible.sync="GRADdialogVisible"
                    width="30%">
                    <span> <el-descriptions title="Benchmark Information" border size="small">
                            <!-- Algo, model and dataset-->
                            <el-descriptions-item label="Algortihm"  span="6">
                                <el-tag effect="dark" :hit="true" :color="gradbenchmarkdata.algorithm === 'SEAM' ?'#00bcd4':'#ffc107'"  
                                style="font-weight: bolder;" >
                                    {{ gradbenchmarkdata.algorithm }}</el-tag></el-descriptions-item>
                            
                            <el-descriptions-item label="Model" span="2">{{gradbenchmarkdata.modelFile}}</el-descriptions-item>
                            <el-descriptions-item label="Dataset" span="2">{{ gradbenchmarkdata.datasetFile }}</el-descriptions-item>         
                        
                            <!-- Epoch(Grad) -->
                            <el-descriptions-item v-if="gradbenchmarkdata.algorithm === 'GradSplitter'" 
                                label="Epochs(Head)" span="2">{{ gradbenchmarkdata.epochs_head }}</el-descriptions-item>
                            <el-descriptions-item v-if="gradbenchmarkdata.algorithm === 'GradSplitter'" 
                                label="Epochs(Modularity)" span="2">{{ gradbenchmarkdata.epochs_modularity }}</el-descriptions-item>
                                                      
                            <!-- learningRate(Both) -->
                            <el-descriptions-item label="Learning rate(Head)"  span="2">{{ gradbenchmarkdata.learningRate_head }}</el-descriptions-item>
                            <el-descriptions-item label="Learning rate(Modulartiy)"  span="2">{{ gradbenchmarkdata.learningRate_modularity }}</el-descriptions-item>
                        </el-descriptions> </span>
                    <span> <el-input readonly resize="none"
                        type="textarea"
                        rows="7"
                        v-model="gradbenchmarklogs"
                        style="width: 100%;margin-top: 40px;">
                    </el-input></span>
                    <span slot="footer" class="dialog-footer">
                        <el-button type="primary" @click="openGRADdialog">RUN</el-button>
                        <el-button type="warning" @click="GRADdialogVisible = false">Close</el-button>
                    </span>
                </el-dialog>
            </div>
        </el-main>
    </el-container>
</template>


<script>
import axios from 'axios';
import io from 'socket.io-client';

export default {
    created(){
        // 初始化socket连接
        this.socket = io('http://localhost:5000/');

        // 设置socket事件监听器
        this.socket.on('connect', () => {
            console.log('socket connected');
        });


        // SEAM SOCKET ON 
        this.socket.on('seam_result', (data) => {
            console.log('received seam result: ' + JSON.stringify(data));
            this.seambenchmarklogs += 'SeaM Result: ' + JSON.stringify(data) + '\n';
        });

        this.socket.on('seam_message', (data) => {
            console.log('received seam message: ' + data);
            this.seambenchmarklogs += 'Message: ' + data + '\n';
        });

        // GRAD SOCKET ON 
        this.socket.on('grad_result', (data) => {
            console.log('received grad result: ' + JSON.stringify(data));
            this.gradbenchmarklogs += 'GradSplitter Result: ' + JSON.stringify(data) + '\n';
        });

        this.socket.on('grad_message', (data) => {
            console.log('received grad message: ' + data);
            this.gradbenchmarklogs += 'Message: ' + data + '\n';
        });

    },
    beforeDestroy() {
    // 在组件销毁前，移除事件监听器并关闭socket连接
        this.socket.off('connect');
        this.socket.off('seam_result');
        this.socket.off('seam_message');
        this.socket.off('grad_result');
        this.socket.off('grad_message');
        this.socket.close();
  },
    data () {
        return{
            SEAMdialogVisible :false,
            GRADdialogVisible: false,
            seambenchmarklogs:'',
            gradbenchmarklogs:'',
            dataTable:{modelName:'SimCNN', dataset:'CIFAR-10', epochs:'100',lr:'0.01', alogrithm:'GradSplitter'},
            tableDataSEAM: [{
                targetProblem: 'Binary Classification',
                modelName: 'VGG16-CIFAR10',
                learningRate: '0.01',
                alpha:'1.00', targetClass:'0'
            }, {
                targetProblem: 'Binary Classification',
                modelName: 'VGG16-CIFAR100',
                learningRate: '0.05',
                alpha:'1.50', targetClass:'0'
            }, {
                targetProblem: 'Binary Classification',
                modelName: 'ResNet20-CIFAR10',
                learningRate: '0.05',
                alpha:'1.00', targetClass:'0'
            }, {
                targetProblem: 'Binary Classification',
                modelName: 'ResNet20-CIFAR100',
                learningRate: '0.12',
                alpha:'1.50', targetClass:'0'
            }, {
                targetProblem: 'Multi-class Classification',
                modelName: 'ResNet20-CIFAR100',
                learningRate: '0.10',
                alpha:'2.00', targetSuperclassIdx:'0' ,targetClass:'0'
            }, {
                targetProblem: 'Multi-class Classification',
                modelName: 'ResNet20-ImageNet',
                learningRate: '0.05',
                alpha:'2.00', targetSuperclassIdx:'0', targetClass:'0'
            }],
            tableDataGrad:[{
                modelName: 'SimCNN-CIFAR10',
                learningRate_head: '0.01',
                learningRate_modularity: '0.001',
                epochs_head:'5',
                epochs_modularity: '140',
            },{
                modelName: 'SimCNN-SVHN',
                learningRate_head: '0.01',
                learningRate_modularity: '0.001',
                epochs_head:'5',
                epochs_modularity: '140',
            },{
                modelName: 'ResCNN-CIFAR10',
                learningRate_head: '0.01',
                learningRate_modularity: '0.001',
                epochs_head:'5',
                epochs_modularity: '140',
            },{
                modelName: 'ResCNN-SVHN',
                learningRate_head: '0.01',
                learningRate_modularity: '0.001',
                epochs_head:'5',
                epochs_modularity: '140',
            } ,{
                modelName: 'InceCNN-CIFAR10',
                learningRate_head: '0.01',
                learningRate_modularity: '0.001',
                epochs_head:'5',
                epochs_modularity: '140',
            },{
                modelName: 'InceCNN-SVHN',
                learningRate_head: '0.01',
                learningRate_modularity: '0.001',
                epochs_head:'5',
                epochs_modularity: '140',
            }],
            seambenchmarkdata: {algorithm:'SEAM', //test Data SEAM
                modelFile:'VGG16', datasetFile:'CIFAR-10', learningRate:'0.01',
                directModelReuse:'Binary Classification', alpha:'1.00' ,targetSuperclassIdx:'2', targetClass:'0'},
            gradbenchmarkdata: {algorithm:'GradSplitter', //test Data Grad
                modelFile:'SimCNN', datasetFile:'CIFAR-10',learningRate_head:'0.01',learningRate_modularity:'0.001',
                epochs_head:'5',epochs_modularity:'140' },    
            
        }

    },
    methods: {
        //路由
        jumptohome(){
            this.$router.push('/modularization')
        },

        // 调整表格格式方法
        coloredrow({row, rowIndex}) {
            if (rowIndex === 0) {
             return 'success-row';
            } 
            return '';
        },
        objectSpanMethod({ row, column, rowIndex, columnIndex }) {
            if (columnIndex === 0) {
                if(rowIndex == 0){return {rowspan:4, colspan:1}}
                else if(rowIndex == 4){return {rowspan:2, colspan:1}}
                return  {rowspan:0, colspan:0}
            }
        },
        renderheader(h, { column }) { // renderheader函数得用到el-table-column上，而不是el-table
            return h("span", {}, [
                h("span", {}, column.label.split("//")[0]), // 其中//也可以用其他符号替代
                h("br"),
                h("span", {}, column.label.split("//")[1]),
            ]);
        },

        // 弹窗中与后段沟通方法
        openSEAMdialog(){
            const data = {
                modelFile: 'vgg16',
                datasetFile: 'cifar10',
                algorithm: 'SEAM',
                learningRate: this.seambenchmarkdata.learningRate,
                directModelReuse: this.seambenchmarkdata.directModelReuse,
                targetClass: this.seambenchmarkdata.targetClass,
                alpha: this.seambenchmarkdata.alpha,
                epoch: '',
                targetSuperclassIdx: '',
           };  
           axios.post('http://localhost:5000/benchmark', data)
                .then(response => {})
                .catch(error => {
                    // return errors
                    console.error(error);
                    this.seambenchmarklogs = 'An error occurred while deloping the benchmark.';
            });
        },
        openGRADdialog(){
            const data = {
                modelFile: 'simcnn',
                datasetFile: 'cifar10',
                algorithm: 'GradSplitter',
                learningRate: '',
                directModelReuse: '',
                targetClass: '',
                alpha: '',
                epoch: '',
                targetSuperclassIdx: '',
           };  
           axios.post('http://localhost:5000/benchmark', data)
                .then(response => {})
                .catch(error => {
                    // return errors
                    console.error(error);
                    this.gradbenchmarklogs = 'An error occurred while deloping the benchmark.';
            });
        },
    }
}



</script>

<style>
.el-container {
    height: 100%; 
    width: 100%;
    font-family: Arial, sans-serif;
}
.el-header, .el-footer {
    background-color: #b2dfdb;
    color: #333;

    text-align: left;
    font-size: x-large;
    font-weight: bolder;
    line-height: 90px;
}
.el-main {
    /* background-color: #E9EEF3; */
    color: #333;
    /* line-height: 20px; */

}
.el-form-item {
    margin-right: 100px;
}
.form-body {
    margin-top: 60px;
}
.selectItem .el-form-item__label{
    font-size: large;
    font-weight: bold;
}
.direct-model-reuse .el-form-item__content{
    font-size: large;
}
.direct-model-reuse .el-radio{
    font-size: large;
}
.modelDataset .el-radio /deep/ .el-radio__label{
  min-width: 100px;
}

.tableBody .el-table .cell {
  white-space: pre-line;
}
.tableBody .el-table .success-row {
    font-weight: bolder;
  }

</style>