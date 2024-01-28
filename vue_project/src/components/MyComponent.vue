<template>
  <div class="container">
    <div class="tabs">
      <button class="tab-button" @click="activeTab = 'modularization'">Modularization</button>
      <button class="tab-button" @click="activeTab = 'moduleDeployment'">Module Deployment</button>
    </div>

    <!-- Modularization tab -->
    <div v-if="activeTab === 'modularization'" class="form">
      <h2>Modularization</h2>
      <div class="row">
        <!-- This column is in left side for Drop-down menu -->
        <div class="column column-select">
          <div class="input-group">
            <label for="algorithm" class="input-label">Select Algorithm</label>
            <select id="algorithm" v-model="algorithm" class="input-select">
              <option disabled value="">Please select an algorithm</option>
              <option>SEAM</option>
              <option>GradSplitter</option>
              <!-- Add other algorithms here -->
            </select>
          </div>
          <div class="input-group" v-if="algorithm === 'SEAM'">
            <label for="direct-model-reuse" class="input-label">Direct Model Reuse</label>
            <select id="direct-model-reuse" v-model="directModelReuse" class="input-select">
              <option disabled value="">Please select a mode</option>
              <option>Binary Classification</option>
              <option>Multi-Class Classification</option>
            </select>
          </div>
          <div class="input-group" v-if="directModelReuse === 'Multi-Class Classification'">
            <label for="target-superclass-idx" class="input-label">Target Superclass Idx</label>
            <select id="target-superclass-idx" v-model="targetSuperclassIdx" class="input-select">
              <option disabled value="">Please select a target superclass index</option>
              <option>0</option>
              <option>1</option>
              <option>2</option>
              <option>3</option>
              <option>4</option>
            </select>
          </div>
          <div v-if="directModelReuse === 'Binary Classification'" class="input-group">
            <label for="target-class" class="input-label">Target Class</label>
            <select id="target-class" v-model="targetClass" class="input-select">
              <option disabled value="">Please select a target class</option>
              <option>0</option>
              <option>1</option>
            </select>
          </div>
          <div class="input-group">
            <label for="model-file" class="input-label">Model File</label>
            <div class="input-with-button">
              <button @click="modelFileUploadMode = modelFileUploadMode === '0' ? '1' : '0'">
                Switch
                <!-- {{ modelFileUploadMode === '0' ? 'Switch to select from list' : 'Switch to upload file' }} -->
              </button>
              <input v-if="modelFileUploadMode === '0'" id="model-file" type="file" @change="onModelFileChange" />
              <select v-else id="model-file" v-model="modelFile">
                <option disabled value="">Please select a model</option>
                <option v-for="model in modelFileOptionsForMultiClass" :key="model">{{ model }}</option>
              </select>
            </div>
          </div>
          <div class="input-group" v-if="modelFile">
            <label for="dataset-file" class="input-label">Dataset File</label>
              <div class="input-with-button">
              <button @click="datasetFileUploadMode = datasetFileUploadMode === '0' ? '1' : '0'">
                  Switch
                <!-- {{ datasetFileUploadMode === '0' ? 'Switch to select from list' : 'Switch to upload file' }} -->
              </button>
              <input v-if="datasetFileUploadMode === '0'" id="dataset-file" type="file" @change="onDatasetFileChange" />
              <select v-else id="dataset-file" v-model="datasetFile">
                <option disabled value="">Please select a dataset</option>
                <option v-for="dataset in datasetFileOptionsForMultiClass" :key="dataset">{{ dataset }}</option>
              </select>
            </div>
          </div>

        </div>
        <!-- This column is in right side for inputs -->
        <div class="column column-input">
          <div v-if="algorithm !== 'SEAM'" class="input-group">
              <label for="epoch" class="input-label">Epoch</label>
              <input id="epoch" type="number" v-model="epoch" step="1" placeholder="Enter epoch" min="1" />
              <div v-if="!isEpochValid" class="error-message">Integer.</div>
          </div>
          <div class="input-group">
              <label for="learning-rate" class="input-label">{{ algorithm === 'SEAM' ? 'Learning Rate Mask' : 'Learning Rate' }}</label>
              <input id="learning-rate" type="number" v-model="learningRate" step="0.001" placeholder="Enter learning rate" min="0.001" />
              <div v-if="!isLearningRateValid" class="error-message">Non-negative.</div>
          </div>
          <div class="input-group" v-if="algorithm === 'SEAM'">
            <label for="alpha" class="input-label">Alpha</label>
            <input id="alpha" type="number" v-model="alpha" step="0.01" placeholder="Enter alpha" min="0" />
            <div v-if="!isAlphaValid" class="error-message">Alpha must be non-negative.</div>
          </div>
        </div>
      </div>
      <div class="button-group">
      <button class="run-button" @click="run">Run</button>
      <!-- Download Button with checking -->
      <button class="download-button" :disabled="!isModelReady" @click="download">Download Processed Model</button>
      </div>
      <textarea readonly v-model="logs" class="logs" />
      <div v-if="modelResult">
        Model Result: {{ modelResult.status }}
      </div>
        <!-- <p>Message: {{ message }}</p>
        <p>Model Status: {{ modelStatus }}</p> -->
    </div>

    <!-- Module Deployment tab -->
    <div v-if="activeTab === 'moduleDeployment'">
      <h2>Module Deployment</h2>
      <!-- Copy the content of Modularization here -->
    </div>
  </div>
</template>

<script>
import axios from 'axios';

// Vue.use(new VueSocketIO({
//   debug: true,
//   connection: 'http://127.0.0.1:5000' // Flask URL
// }))

export default {
  sockets:{
  connect: function(){
    console.log('socket connected')
  },
  model_result: function(data){
    console.log('received model result: ' + JSON.stringify(data));
    this.modelResult = data;
    this.logs += 'Model Result: ' + JSON.stringify(data) + '\n';
  },
  message: function(data){
    console.log('received message: ' + data);
    this.logs += 'Message: ' + data + '\n';
  }
},

  data() {
    return {
      activeTab: 'modularization',  // Currently active tab
      modelFile: null,  // The model file selected by the user
      datasetFile: null,  // The dataset file selected by the user
      algorithm: '',  // The algorithm selected by the user
      epoch: '1',  // The epoch entered by the user
      learningRate: '',  // The learning rate entered by the user
      logs: '',  // Running logs
      isModelReady: false,  // Whether the model is ready to be downloaded
      modelFileUploadMode: '1',  // The upload mode for the model file (0: file upload, 1: select from list)
      datasetFileUploadMode: '1',  // The upload mode for the dataset file (0: file upload, 1: select from list)
      directModelReuse: '', //To save the choose of Direct model reuse
      targetClass: '-1',  // The target class selected by the user
      alpha: '1',  // The alpha value entered by the user, default to 1
      targetSuperclassIdx: '-1',  // The target superclass index selected by the user

      message: '',
      modelStatus: '',
      modelResult: null,
    };
  },
  // mounted () {
  //   this.$socket.client.on('message', (data) => {
  //     this.message = data
  //   })
  //   this.$socket.client.on('model_result', (data) => {
  //     if (data.status == 'success') {
  //       this.modelStatus = "Model run successfully"
  //     } else {
  //       this.modelStatus = "Model run failed: " + data.error
  //     }
  //   })
  // },

  computed: {
    // To make sure multi-class has the two model
    modelFileOptionsForMultiClass() {
      return this.directModelReuse === 'Multi-Class Classification' ? ['ResNet20', 'ResNet50'] : this.modelFileOptions;
    },
    // To make sure models corresponds to their datasets
    datasetFileOptionsForMultiClass() {
    if (this.directModelReuse === 'Multi-Class Classification') {
      if (this.modelFile === 'ResNet20') {
        return ['CIFAR100'];
      } else if (this.modelFile === 'ResNet50') {
          return ['ImageNet'];
        }
      }
    return this.datasetFileOptions;
    },
    isEpochValid() {
      const epochNumber = Number(this.epoch);
      return Number.isInteger(epochNumber) && this.epoch > 0;
    },
    isLearningRateValid() {
      return this.learningRate > 0;
    },
    modelFileOptions() {
    return this.algorithm === 'SEAM' ? ['vgg16', 'resnet20'] : ['Model 1', 'Model 2', 'Model 3'];
    },
    datasetFileOptions() {
      return this.algorithm === 'SEAM' ? ['cifar10', 'cifar100'] : ['Dataset 1', 'Dataset 2', 'Dataset 3'];
    },
    isAlphaValid() {
      return this.alpha >= 0;
    },
  },

  methods: {
    // The handler function when the user selects a model file
    onModelFileChange(event) {
      this.modelFile = event.target.files[0];
    },
    // The handler function when the user selects a dataset file
    onDatasetFileChange(event) {
      this.datasetFile = event.target.files[0];
    },
    // The handler function when the user clicks the run button
    run() {
      if (!this.modelFile) {
        window.alert('Please select a model file.');
        return;
      }

      if (!this.datasetFile) {
        window.alert('Please select a dataset file.');
        return;
      }

      if (!this.algorithm) {
        window.alert('Please select an algorithm.');
        return;
      }

      if (!this.isEpochValid) {
        window.alert('Integer');
        return;
      }

      if (!this.isLearningRateValid) {
        window.alert('Non-negative.');
        return;
      }

      // If all conditions are satisfied, continue running the model...
      window.alert('Running...');
      // Here execute the running code of the model, you may need to use Axios or other HTTP libraries to interact with the back-end server
      // When the server returns logs, you can add them to this.logs

      // If all conditions are satisfied, continue running the model...
        // After running the model, set isModelReady to true
        this.isModelReady = true;

        const data = {
        modelFile: this.modelFile,
        datasetFile: this.datasetFile,
        algorithm: this.algorithm,
        epoch: this.epoch,
        learningRate: this.learningRate,
        directModelReuse: this.directModelReuse,
        targetClass: this.targetClass,
        alpha: this.alpha,
        targetSuperclassIdx: this.targetSuperclassIdx,
      };

      // Send POST requests to Flask
      axios.post('http://127.0.0.1:5000/run_model', data)
        .then(response => {
          // success, return results
          this.logs = response.data.logs;
          this.isModelReady = response.data.isModelReady;
        })
        .catch(error => {
          // return errors
          console.error(error);
          this.logs = 'An error occurred while running the model.';
        });

    },
      download() {
      axios({
        method: 'GET',
        url: 'http://127.0.0.1:5000/download',  // Flask后端下载路由
        responseType: 'blob'  // 表明返回类型是 Blob
      })
      .then(response => {
        // 创建一个不可见的 'a' 标签用于下载
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'lr_head_mask_0.1_0.01_alpha_1.0.pth');  // 你的文件名
        document.body.appendChild(link);
        link.click();
      })
      .catch(err => {
        console.log(err);
      });
    },  
  },
};
</script>

<style>
.container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 2em;
  font-family: Arial, sans-serif;
}
.tabs {
  margin-bottom: 1em;
}
.tab-button {
  font-size: 1.2em;
  padding: 0.5em 1em;
  margin-right: 1em;
  border-radius: 12px;
  border: 1px solid black;  /*Add black border */
  background-color: #d8d8d8; /* Gray */
  color: rgb(0, 0, 0);
  cursor: pointer;
  transition: background-color 0.3s;
}
.tab-button:hover {
  background-color: #9f9f9f; /* Darker Gray */
}
.form {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}
.row {
  display: flex;
  margin-bottom: 1em;
  /* 靠右 */
  justify-content: flex-end; 
}
.column {
  display: flex;
  flex-direction: column;
  margin-right: 2em;
}
.column-select, .column-input {
  flex: 1;  /* 让两列各自占用可用空间的一半 */
}

.column-select {
  flex: none; /* 移除 flex: 1; */
  width: 45%; /* 固定宽度 */
}

.column-input {
  flex: none; /* 移除 flex: 1; */
  width: 45%; /* 固定宽度 */
}

.input-with-button {
  display: flex;
  align-items: center; /* 垂直居中对齐 */
}

.input-with-button button {
  flex-shrink: 0; /* 防止按钮缩小 */
  margin-right: 10px; /* 右边距 */
}

.input-with-button input[type="file"],
.input-with-button select {
  flex-grow: 1;  /* 允许输入框增长 */
  min-width: 0;  /* 设置一个最小宽度 */
}

.input-group {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1em;
  width: 100%;  /* Or set a fixed width */
}

.input-group > label {
  font-weight: bold;
  margin-bottom: .5em;
}
.button-group {
  display: flex;
  justify-content: space-between;
  width: 50%;
  margin: 0 auto;
}
.run-button {
  padding: 1em;
  font-size: 1.2em;
  margin-bottom: 1em;
  border-radius: 12px;
  border: 1px solid black;  /* Add black border */
  background-color: #d8d8d8; /* Gray */
  color: rgb(0, 0, 0);
  cursor: pointer;
  transition: background-color 0.3s;
}
.run-button:hover {
  background-color: #9f9f9f; /* Darker Gray */
}
.logs {
  width: 100%;
  min-height: 200px;  /* Make the logs textarea taller */
  margin-bottom: 1em;
  border-radius: 12px;  /* Make the logs textarea rounded */
  border: 1px solid black;  /* Add black border*/
}

.download-button {
  /* ... */
  padding: 1em;
  font-size: 1.2em;
  margin-bottom: 1em;
  border-radius: 12px;
  border: 1px solid black;  /* Add black border */
  background-color: #d8d8d8; /* Gray */
  color: rgb(0, 0, 0);
  cursor: pointer;
  transition: background-color 0.3s;
}
.download-button:hover {
  background-color: #9f9f9f; /* Darker Gray */
}
.download-button:disabled {
  background-color: #ffffff; /* Gray */
  cursor: not-allowed;
  opacity: 0.6;
}

</style>