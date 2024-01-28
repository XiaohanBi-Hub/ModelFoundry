<template>
    <el-container>
    <!-- Header -->
        <el-header style="height: 85px;"><span style="margin-top: 0;margin-left: 5%;"  @click="jumptohome"> 
                ModelFoundry
        </span></el-header>

        <el-main>
            <h2 style="text-align: left;margin-left: 10.5%;margin-top: 2%;margin-bottom: 3%;">
                <i class="el-icon-arrow-left" style="margin-right: 5px;" @click="jumptohome"></i>
                Module Reuse</h2>

            <!-- Main body -->
            <div style="text-align: center;">
                <!-- Data Tabel--> 
                <div class="DatatTable" style="margin-left: 14%;margin-right: 15%;margin-top: 40px;">
                    <el-descriptions title="Model Modularization Information" border>
                        <!-- Algorithm, model and dataset-->
                        <el-descriptions-item label="Algortihm" labelStyle="width:15%" contentStyle="width:15%">
                            <el-tag effect="dark" :hit="true" :color="dataTable.algorithm === 'SEAM' ?'#00bcd4':'#ffc107'" 
                            style="font-weight: bolder;" >
                                {{ dataTable.algorithm }}</el-tag></el-descriptions-item>
                        <el-descriptions-item label="Model" labelStyle="width:13%" contentStyle="width:17%">{{dataTable.modelFile}}</el-descriptions-item>
                        <el-descriptions-item label="Dataset" labelStyle="width:13%" contentStyle="width:17%">{{ dataTable.datasetFile }}</el-descriptions-item>
                    
                        <!-- Epoch(Grad) -->
                        <el-descriptions-item v-if="dataTable.algorithm === 'GradSplitter'" 
                            label="Epochs">{{ dataTable.epoch }}</el-descriptions-item>
                        
                        <!-- Target Class(SEAM) -->
                        <el-descriptions-item v-if="dataTable.algorithm === 'SEAM'" span="6"
                            label="Model Reuse Method">{{ dataTable.directModelReuse }}</el-descriptions-item>
                        <el-descriptions-item v-if="dataTable.algorithm === 'SEAM' && dataTable.directModelReuse === 'Multi-Class Classification'"
                            label="Target Superclass Idx">{{ dataTable.targetSuperclassIdx }}</el-descriptions-item>
                        <el-descriptions-item v-if="dataTable.algorithm === 'SEAM' && dataTable.directModelReuse === 'Binary Classification'"
                            label="Target Class Idx">{{ dataTable.targetClass }}</el-descriptions-item>
                        
                        <!-- learningRate(Both) -->
                        <el-descriptions-item label="Learning rate">{{ dataTable.learningRate }}</el-descriptions-item>
                        
                        <!-- alpha(SEAM) -->
                        <el-descriptions-item v-if="dataTable.algorithm === 'SEAM'" 
                            label="Alpha">{{ dataTable.alpha }}</el-descriptions-item>
                    </el-descriptions>
                </div>

                <!-- Upload pic and deployment-->
                <h4 style="text-align: left;margin-left: 14%;margin-right: 15%;margin-top: 80px;">
                    Module Deployment</h4>
                <div style="margin-left: 14%;margin-right: 15%;margin-top: 20px;text-align: left;">
                    <el-row> <el-col :span="8">
                        <div class="figfileList" >
                            <el-card class="catdogcard" :body-style="{ padding: '10px 20px' }" 
                            :shadow="(this.selectedImage === 'cat')? 'always' : 'hover'">
                                <el-image :src="catURL" style="height: 80px;width:100px;float:left" fit ="cover"></el-image>
                                <el-radio v-model="selectedImage" label="cat" >Cat </el-radio>
                            </el-card>

                            <el-card class="catdogcard" :body-style="{ padding: '10px 20px' }"
                            :shadow="(this.selectedImage === 'dog')? 'always' : 'hover'">
                                <el-image :src="dogURL" style="height: 80px;width:100px;float:left" fit ="cover"></el-image>
                                <el-radio v-model="selectedImage" label="dog" >Dog </el-radio>
                            </el-card>

                            <el-button type="primary"  slot="trigger" style="margin-bottom: 10px;float: right;" disabled=true> 
                                Browse and Upload Image </el-button>
                            <el-button style="margin-left: 10px;margin-bottom: 10px;" type="warning" 
                                @click="submitDelopyment">Run</el-button>
                        </div> </el-col>

                        <el-col :span="16"><div style="text-align: right;">
                            <el-input readonly resize="none"
                                type="textarea"
                                rows="12"
                                v-model="deploymentlogs"
                                style="width: 96%;margin-right: 0px;">
                            </el-input>
                        </div></el-col>
                    </el-row>
                </div>
            </div>
        </el-main>
    </el-container>
 
</template>


<script>
import axios from 'axios';
import io from 'socket.io-client';
import catfig from '@/image/cat.jpg'
import dogfig from '@/image/dog.jpg'
export default {
    created (){
        var modelFile = sessionStorage.getItem("modelFile");
        var datasetFile = sessionStorage.getItem("datasetFile");
        var algorithm = sessionStorage.getItem("algorithm");
        var epoch = sessionStorage.getItem("epoch");
        var learningRate = sessionStorage.getItem("learningRate");
        var directModelReuse = sessionStorage.getItem("directModelReuse");
        var targetClass = sessionStorage.getItem("targetClass");
        var targetClassLabel = sessionStorage.getItem("targetClassLabel");
        var alpha = sessionStorage.getItem("alpha");
        var targetSuperclassIdx = sessionStorage.getItem("targetSuperclassIdx");
        var targetSuperclassLabel = sessionStorage.getItem("targetSuperclassLabel");
        this.dataTable.modelFile = modelFile
        this.dataTable.datasetFile = datasetFile
        this.dataTable.algorithm = algorithm
        this.dataTable.epoch = epoch
        this.dataTable.learningRate = learningRate
        this.dataTable.directModelReuse = directModelReuse
        this.dataTable.targetClass = targetClassLabel
        this.dataTable.alpha = alpha
        this.dataTable.targetSuperclassIdx = targetSuperclassLabel
        console.log(this.dataTable)

        // 初始化socket连接
        this.socket = io('http://127.0.0.1:5000/');

        // 设置socket事件监听器
        this.socket.on('connect', () => {
            console.log('socket connected');
        });

        this.socket.on('deployment_result', (data) => {
            console.log('received deployment result: ' + JSON.stringify(data));
            this.deploymentlogs += 'Predict Result: ' + JSON.stringify(data) + '\n';
        });

        this.socket.on('deployment_message', (data) => {
            console.log('received deployment message: ' + data);
            this.deploymentlogs += 'Message: ' + data + '\n';
        });

    },
    beforeDestroy() {
    // 在组件销毁前，移除事件监听器并关闭socket连接
        this.socket.off('connect');
        this.socket.off('deployment_result');
        this.socket.off('deployment_message');
        this.socket.close();
    },
    data () {
        return{
            selectedImage: '', // ='cat' or 'dog'
            dataTable: {
                algorithm: '', // = 'GradSplitter' or 'SEAM'; The algorithm selected by the user
                modelFile: null,  // The model file selected by the user
                datasetFile: null,  // The dataset file selected by the user
                learningRate: 0.01,  // The learning rate entered by the user
                epoch: 145,  // The epoch entered by the user
                alpha: 1,  // The alpha value entered by the user, default to 1
                directModelReuse: '', //To save the choose of Direct model reuse
                targetSuperclassIdx: '',  // The target superclass index selected by the user
                targetClass: '',  // The target class selected by the user
            },
            fileList:[],

            deploymentlogs: '',  // Running logs 
            imageUrl: 'https://t7.baidu.com/it/u=4162611394,4275913936&fm=193&f=GIF' ,
            catURL: catfig,
            dogURL: dogfig,
        }

    },
    methods: {
        jumptohome(){
            this.$router.push('/modularization')
        },
        submitDelopyment(){
            const data = {
                image: this.selectedImage
            };
            axios.post('http://127.0.0.1:5000/run_deployment', data)
            .catch(error => {
                // return errors
                console.error(error);
                this.logs = 'An error occurred while running the model.';
            });
        },  
    },
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
.el-tag--dark.is-hit {
    border:0;
}
.el-upload --text {
    align-items: left;
}

.catdogcard {
    height:100px;
    margin-bottom: 10px;
    line-height: 80px;
    text-align: center;
}
</style>