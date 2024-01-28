<template>
    <el-container>
      <!-- Header -->
      <el-header style="height: 85px;"> <span style="margin-top: 0;margin-left: 5%;"> 
        ModelFoundry
      </span> </el-header>
  
      <!-- Page Body-->
      <el-container>
        <!-- Main body -->
        <el-aside style="width: 78%;border-width:0 1px 0 0;border-style: solid; border-color: #E4E7ED;">
          <!-- Title -->
          <h2 style="text-align: left;margin-left: 15%;float: left;margin-top: 2%;margin-bottom: 3%;">
                Model Modularization</h2>
          
          <el-button style="float:right;margin-right: 5%;margin-top: 2%;font-size: larger;width:220px;margin-bottom: 10px;" 
                      type="success" @click="JumpToBenchmark">
                    Benchmark</el-button>
          
          <!-- Form -->
          <div class="form-body" style="margin-left: 20%;margin-right: 5%;margin-top: 100px;">
              <el-form  label-width="220px" label-position="left" 
              ref="form" :model="form" id="selectForm" :inline="true" >
                  <!-- Model Selection-->
                  <el-form-item label="Model File:" class="selectItem" style="width: 100%;">
                    <el-select v-model="modelFile" placeholder="Please select a model" filterable @change="AlgorithSelectHelperFromSelection(fromSelection)">
                    <!-- <el-select v-model="modelFile" placeholder="Please select a model" @change="ResetAll"> -->
                      <el-option-group v-for="group in wholeModelFileOptions" :key="group.label" :label="group.label">
                          <el-option  v-for="item in group.options" 
                          :key="item.value" :label="item.label" :value="item.value" :disabled="item.disabled">
                          </el-option></el-option-group>  
                      </el-select>
                  </el-form-item>
  
                  <!-- Dataset Selection -->
                  <el-form-item label="Dataset File:" class="selectItem" style="width: 100%;">
                    <el-select v-model="datasetFile" placeholder="Please select a dataset" @change="AlgorithSelectHelper" :disabled="fromSelection==='true'?true:false">
                        <el-option  v-for="item in wholeDatasetFileOptions" 
                        :key="item.value" :label="item.label" :value="item.value">
                          </el-option>
                      </el-select>
                  </el-form-item>    
              
                  <!-- Algorithm Select-->
                  <el-form-item v-if="datasetFile!=null && modelFile!=null" label="Algorithm:" class="selectItem" style="width: 100%;"> 
                      <el-select v-model="algorithm" placeholder="Please select an algorithm" >
                          <el-option label="SeaM" value="SEAM" :disabled="selectDisabled.SEAM">  </el-option>
                          <el-option label="GradSplitter" value="GradSplitter" :disabled="selectDisabled.GradSplitter">  </el-option>
                      </el-select>
                  </el-form-item>
                  
                  <!-- ModelReuseMethod(SEAM ONLY) -->
                  <el-form-item v-if="algorithm === 'SEAM'" 
                  label="Model Reuse Method:" class="selectItem" style="width: 100%;">  
                      <div class="chooseFromCards" style="width: 600px"> 
                        <el-row >
                          <el-col :span="12"><div >
                            <el-card style="width: 90%;height: 120px;" :body-style="{ padding: '0px 20px' } " 
                            :shadow="(directModelReuse === 'Multi-Class Classification' || directModelReuse === 'Binary Classification' )? 'always' : 'hover'">
                              <p style="font-weight: bolder;margin-top: 0;margin-bottom:15px;color:#606266;">Direct Model Reuse</p>
                              <el-radio-group v-model="directModelReuse" @change="settargetIdxOptions">
                                <el-radio label="Binary Classification" size="medium" style="margin-bottom: 15px;" :disabled="selectDisabled.binary"> Binary Classification</el-radio>
                                <el-radio label="Multi-Class Classification" size="medium" style="margin-bottom: 15px;" :disabled="selectDisabled.multi"> Multi-Class Classification</el-radio>
                              </el-radio-group>
                            </el-card>
                          </div></el-col>
                          <el-col :span="12"><div >
                            <!-- <el-card style="width: 90%;height: 120px;" :body-style="{ padding: '0px 20px' }"  
                            :shadow="(this.directModelReuse === 'Defect Inheritance')? 'always' : 'hover'">
                              <p style="font-weight: bolder;margin-top: 0;margin-bottom:15px;color:#606266">Indirect Model Reuse</p>
                              <el-radio-group v-model="directModelReuse">
                                <el-radio label="Defect Inheritance" size="medium" style="margin-bottom: 15px;" :disabled="selectDisabled.defect">Defect Inheritance</el-radio>
                              </el-radio-group>
                            </el-card> -->
                          </div></el-col>
                        </el-row>
                      </div>
                  </el-form-item>
                  
                  <!-- ModelReuseMethod (Grad) -->
                  <el-form-item v-if="algorithm === 'GradSplitter'" 
                  label="Model Reuse Method:" class="selectItem" style="width: 100%;">  
                      <div class="chooseFromCards" style="width: 600px"> 
                        <el-row ><el-col :span="12"><div >
                            <el-card style="width: 90%;height: 120px;" :body-style="{ padding: '0px 20px' } " 
                            :shadow="(reuseMethod === 'More Accurate' || reuseMethod === 'For New Task'  )? 'always' : 'hover'">
                              <p style="font-weight: bolder;margin-top: 0;margin-bottom:15px;color:#606266;">Module Composition</p>
                              <el-radio-group v-model="reuseMethod">
                                <el-radio label="More Accurate" size="medium" style="margin-bottom: 15px;" >More Accurate</el-radio>
                                <!-- <el-radio label="For New Task"  size="medium" style="margin-bottom: 15px;" >For New Task</el-radio> -->
                              </el-radio-group>
                            </el-card>
                          </div></el-col></el-row>
                      </div>
                  </el-form-item>
                  <!-- <el-form-item v-if="algorithm === 'GradSplitter' && reuseMethod==='For New Task'" label="Choose Modules:" class="selectItem">
                    <el-select v-model="cifarclass" placeholder="Modules from CIFAR-10" style="margin-bottom: 10px;margin-right: 20px;">
                      <el-option v-for="item in cifar10classes" :key="item.value" :label="item.label" :value="item.value"></el-option>
                    </el-select> 
                    <el-select v-model="svhnclass" placeholder="Modules from SVHN" >
                      <el-option v-for="item in svhnclasses" :key="item.value" :label="item.label" :value="item.value"></el-option>
                    </el-select>
                    <p style="color: darkgray;margin-top: 0;line-height: 10px;">Build a binary-classification model by combination two modules. </p>
                  </el-form-item> -->
  
                  <!-- targetSuperclassIdx(SEAM+Multi-Class Classification ONLY) -->
                  <el-form-item v-if="algorithm === 'SEAM' && directModelReuse === 'Multi-Class Classification'" 
                  label="Target Superclass Idx:" class="selectItem" style="width: 100%;"> 
                      <el-select v-model="targetSuperclassIdx" class="selectLong"
                      placeholder="Please select a target superclass index" >
                          <el-option  v-for="item in targetSuperclassIdxOptions" 
                          :key="item.value" :label="item.label" :value="item" :disabled = "item.disable">
                          </el-option>
                      </el-select>
                  </el-form-item>
  
                  <!-- targetClass(SEAM+Binary Classification) ONLY -->
                  <el-form-item v-if="algorithm === 'SEAM' && directModelReuse === 'Binary Classification'" 
                  label="Target Class Idx:" class="selectItem" style="width: 100%;">
                      <el-select v-model="targetClass" :disabled="targetClass.label != ''? true:false"
                      placeholder="Please select a target class">
                          <el-option  v-for="item in targetClassIdxOptions" 
                          :key="item.value" :label="item.label" :value="item">
                          </el-option>
                      </el-select>
                  </el-form-item> 
  
                  <!-- Epoch / alpha / LearningRate  -->
                  <!-- EPOCH(GradSplitter) -->
                  <el-form-item v-if="algorithm === 'GradSplitter'" label="Epochs:" class="selectItem" style="margin-right: 5%;">  
                    <el-tooltip class="item" effect="dark" content="The number of epochs must be an integer." placement="bottom-start">     
                     <el-input-number :disabled="algorithm===''"
                      v-model="epoch" :step="1" :min=1 :max="500" step-strictly></el-input-number>
                    </el-tooltip>
                  </el-form-item>
                  <!-- ALPHA(SEAM) -->
                  <el-form-item v-if="algorithm === 'SEAM'" label="Alpha:" class="selectItem"  style="margin-right: 5%;"> 
                    <el-tooltip class="item" effect="dark" content="Alpha must be non-negative." placement="bottom-start">                   
                      <el-input-number :disabled="algorithm===''"
                      v-model="alpha" :precision=2 :step="0.01" :min=0 :max="2" step-strictly></el-input-number> 
                    </el-tooltip>  
                  </el-form-item>
                  <!-- LEARNING RATE(SEAM) -->
                  <el-form-item v-if="algorithm === 'SEAM' " label="Learning Rate" class="selectItem" > 
                    <el-tooltip class="item" effect="dark" content="Learning rate is usually in the range of (0,1). " placement="bottom-start">     
                      <el-input-number :disabled="algorithm===''"
                      v-model="learningRate" :precision=3 :step="0.001" :min=0.001 :max="1" step-strictly></el-input-number>
                    </el-tooltip>
                  </el-form-item>
              </el-form>
  
              <!-- Buttons -->
              <div style="width: 100%;margin-top: 40px;display: flex; flex-direction: row;flex-wrap: wrap;">
                <el-button style="font-size: larger;margin-bottom: 10px; width: 150px; flex:0 auto;" type="success" 
                  @click="beforeRunModularization">
                Modularize</el-button>
                <el-button style="font-size: larger;margin-bottom: 10px; width: 300px;flex:0 auto;" type="warning" 
                  @click="download"> 
                Download Module</el-button>
                <!-- Download Process Model</el-button> -->
                <el-button v-if="algorithm==='SEAM'" style="justify-content:flex-end;margin-left:auto;font-size: larger;margin-bottom: 10px;flex:0 auto; " type="primary" @click="JumpToDeployment"> 
                Module Reuse </el-button>
                <el-button v-if="algorithm==='GradSplitter'" :disabled="(reuseMethod === 'More Accurate' || reuseMethod === 'For New Task')? false:true" style="justify-content:flex-end;margin-left:auto;font-size: larger;margin-bottom: 10px;flex:0 auto;" type="primary" @click="runReuse"> 
                Module Reuse </el-button>
              </div>
  
              <!-- Result log -->
              <div style="margin-top: 40px;">
                <el-progress v-if="progressrunning" :percentage="progresspersentage" :color="progresscolor" :text-inside="true" :show-text="false" style="width: 100%; " text-color="white"></el-progress>
                <el-input readonly resize="none"
                  type="textarea"
                  rows="7"
                  v-model="logs"
                  style="width: 100%;margin-top: 10px;">
                </el-input>
              </div>
              
          </div>
        </el-aside>
        
        <!-- Right side bar -->
        <el-main style="align-items: center;">
            <el-card style="margin-bottom: 30px;width: 90%;margin-left: 2%;" shadow="never">
              <div slot="header" class="clearfix">
                <span><i class="el-icon-guide"></i></span>
                <span style="font-size: large;font-weight: bold;margin-left: 10px;">Queue</span>
              </div>
  
              <!-- card list-->
              <div >
                <div><el-tag style="margin-right: 10px;color: #333;margin-bottom: 15px;min-width: 190px;font-size: medium; " :color="StatusColor['running']" effect="dark" :hit="true">
                  <i class="el-icon-video-play" style="margin-right: 10px;margin-left: 5px;"></i>
                  <span >Task Running:</span> 
                  <span style="font-weight: bolder;margin-left: 10px;">{{ queueNum.running }} </span>
                </el-tag></div>
                <div><el-tag style="margin-right: 10px;color: #333;margin-bottom: 15px;min-width: 190px;font-size: medium;" :color="StatusColor['pending']" effect="dark" :hit="true">
                  <i class="el-icon-loading" style="margin-right: 10px;margin-left: 5px;"></i>
                  <span >Task Pending:</span> 
                  <span style="font-weight: bolder;margin-left: 10px;">{{ queueNum.pending }} </span>
                </el-tag></div>
                <div><el-tag style="margin-right: 10px;color: #333;margin-bottom: 15px;min-width: 190px;font-size: medium;" :color="StatusColor['done']" effect="dark" :hit="true">
                  <i class="el-icon-finished" style="margin-right: 10px;margin-left: 5px;"></i>
                  <span >Task Done:</span> 
                  <span style="font-weight: bolder;margin-left: 30px;">{{ queueNum.done }} </span>
                </el-tag></div>
              </div>
            </el-card>
            
        </el-main>
      </el-container>
  
      <!-- Dialog:think twice plz (Grad + modularization) -->>
      <el-dialog :visible.sync="AreUSure" width="30%" class="areusuredialog">
        <span slot="title" class="header-title">
          <h2 style="margin-left: 20px;">  <i class="el-icon-warning" style="margin-right: 10px;"></i>
            Warning</h2>
        </span>
        <p style="font-size: medium;">This function will cost around 900 minutes for decomposing 10 pretrained models.
        Please comfirm that you have enough time.</p>
        <div style="text-align: center;height: 50px;margin-top: 40px;">
          <el-button type="primary" @click="runModularization"> I'm Sure </el-button>
          <el-button type="warning" @click="AreUSure = false"> Quit </el-button>
        </div>
      </el-dialog>
    </el-container>
  </template>
  
  <script>
  import axios from 'axios';
  import io from 'socket.io-client';
  export default {
  created(){
    const h = this.$createElement 
    this.$notify({
      title:'Notice',
      message: h('i', {style:'color:red;font-weight:bold;font-size:larger'},
      'Linux and GPU with CUDA is required.'),
      type: 'warning',
      duration: 0,
    })

    // 初始化socket连接
    this.socket = io('http://127.0.0.1:5000/');
  
    // 设置socket事件监听器
    this.socket.on('connect', () => {
      console.log('socket connected');
    });
  
    this.socket.on('model_result', (data) => {
      console.log('received model result: ' + JSON.stringify(data));
      this.logs += 'Model Result: ' + JSON.stringify(data) + '\n';
    });
  
    this.socket.on('message', (data) => {
      console.log('received message: ' + data);
      this.logs += 'Message: ' + data + '\n';
    });
  
    this.socket.on('reuse_result', (data)=>{
      console.log('received reuse result' + JSON.stringify(data)) 
      this.reuselogs += 'Reuse Result: ' + JSON.stringify(data) + '\n';
  
    })
    this.socket.on('reuse_message', (data) => {
      console.log('received message: ' + data);
      this.reuselogs += 'Message: ' + data + '\n';
    });
    this.socket.on('get_progress_percentage', (data) => {
      console.log('received progress data: ' + data);
      this.progresspersentage = parseInt(JSON.stringify(data))
    });
    this.progressrunning = false

    this.fromSelection = sessionStorage.getItem("fromSelection")
    console.log(this.fromSelection)

    if(this.fromSelection === 'true'){
      this.datasetFile = sessionStorage.getItem("datasetFile")
      this.module_tag=sessionStorage.getItem("module_tag")

      if(this.module_tag === 'classes'){
        // targetClass: targetClassLabel: 
        this.targetClass.value = sessionStorage.getItem("module_idx")
        this.targetClass.label = sessionStorage.getItem("module_name")
        // console.log(this.targetClass)
        if(this.datasetFile === 'cifar10'){
          console.log('set targetclass for cifar10')
          this.targetClassIdxOptions= this.cifar10classes
        }
        else if(this.datasetFile === 'cifar100'){
          console.log('set targetclass for cifar100')
          this.targetClassIdxOptions = this.cifar100classes
        }
        else if (this.datasetFile === 'svhn'){
          console.log('set targetclass for svhn')
          this.targetClassIdxOptions = this.svhnclasses
        }
      }else{
        //targetSuperclassIdx
        if(this.datasetFile === 'ImageNet'){
          console.log('set targetSUPERclass for ImageNet')
          this.targetSuperclassIdxOptions = this.imagenetsuperclasses
        }
        else if (this.datasetFile === 'cifar100'){
          console.log('set targetSUPERclass for CIFAR-100')
          this.targetSuperclassIdxOptions = this.cifar100superclasses
        }
        this.targetSuperclassIdx.value = sessionStorage.getItem("module_idx")
        this.targetSuperclassIdx.label = sessionStorage.getItem("module_name")
      }
    }
  },
  beforeMount(){
  },
  mounted(){
    this.timer = setInterval(this.updateProgress, 5000)
  },
  beforeDestroy() {
      // 在组件销毁前，移除事件监听器并关闭socket连接
      this.socket.off('connect');
      this.socket.off('model_result');
      this.socket.off('message');
      this.socket.off('reuse_result');
      this.socket.off('reuse_message');
      this.socket.off('get_progress_percentage');
      this.socket.close();
  
      // clear timer
      clearInterval(this.timer)
    },
  
  data() {
    return {
      fromSelection: 'false', //passed from selection
      module_tag: '',

      selectDisabled:{
        SEAM: false, GradSplitter: false,
        binary:false, multi:false, defect:false,
      },
      form:{},
      modelFile: null,  // The model file selected by the user
      datasetFile: null,  // The dataset file selected by the user
      algorithm: '',  // The algorithm selected by the user
      epoch: 145,  // The epoch entered by the user
      learningRate: 0.01,  // The learning rate entered by the user
      directModelReuse: '', //To save the choose of Direct model reuse
      targetClass: {value:'', label:''},  // The target class selected by the user
      alpha: 1,  // The alpha value entered by the user, default to 1
      targetSuperclassIdx:  {value:'', label:''},  // The target superclass index selected by the user
      
      logs: '',  // Running logs
      reuselogs: '',
      isModelReady: false,  // Whether the model is ready to be downloaded
      modelFileUploadMode: '1',  // The upload mode for the model file (0: file upload, 1: select from list)
      datasetFileUploadMode: '1',  // The upload mode for the dataset file (0: file upload, 1: select from list)
      
      taskid: '',
      taskQueue: {},
      taskStatus: 'running',
      StatusColor: {pending:'#ffd54f', done:'#aed581', running:'#81d4fa'},
      queueNum:{pending:404,running: 0, done:0,},
      lenQueue: 3,
  
      message: '',
      modelStatus: '',
  
      progresspersentage: 0, 
      progresscolor: '#4db6ac',
      progressrunning :false,
      socket: null,
      
      reuseMethod:'More Accurate', // in Model Reuse [=More Accurate/For New Task]
      cifarclass:'',
      svhnclass:'',
      AreUSure:false ,
  
      targetSuperclassIdxOptions:[],
      targetClassIdxOptions:[],
      cifar10classes:[
        {value:"0", label:"0 - Airplane"},{value:"1", label:"1 - Automobile"},
        {value:"2", label:"2 - Bird"},{value:"3", label:"3 - Cat"},
        {value:"4", label:"4 - Deer"},{value:"5", label:"5 - Dog"},
        {value:"6", label:"6 - Frog"},{value:"7", label:"7 - Horse"},
        {value:"8", label:"8 - Ship"},{value:"9", label:"9 - Truck"},
      ],
      svhnclasses:[
        {value:"0", label:"0 - Number 0"},{value:"1", label:"1 - Number 1"},{value:"2", label:"2 - Number 2"},
        {value:"3", label:"3 - Number 3"},{value:"4", label:"4 - Number 4"},
        {value:"5", label:"5 - Number 5"},{value:"6", label:"6 - Number 6"},{value:"7", label:"7 - Number 7"},
        {value:"8", label:"8 - Number 8"},{value:"9", label:"9 - Number 9"},
      ],
      ModelFileList:[
        {value:'vgg16', label:'VGG16', disabled:false},
        {value:'resnet20', label:'ResNet20', disabled:false},
        {value:'resnet50', label:'ResNet50', disabled:false},
        {value:'resnet18', label:'ResNet18', disabled:false},
        {value:'simcnn', label:'SimCNN', disabled:false}, 
        {value:'rescnn', label:'ResCNN', disabled:false}, 
        {value:'incecnn', label:'InceCNN', disabled:false}
      ],
      imagenetsuperclasses: [
        {value:"0", label:"0 - fish(3 classes: tench, goldfish, crampfish)"},
        {value:"1", label:"1 - shark(2 classes: white shark, tiger shark)"},
        {value:"2", label:"2 - bird(5 classes: cock, hen, ostrich, brambling, goldfinch)"},
        {value:"3", label:"3 - salamander(4 classes: salamandra, common newt, eft, spotted salamander)"},
        {value:"4", label:"4 - frog(3 classes: bullfrog, tree frog, tailed frog)"},
        {value:"5", label:"5 - turtle(2 classes: loggerhead, leatherback turtle)"},
        {value:"6", label:"6 - lizard(4 classes: banded gecko, iguana, anole, whiptail)"},
        {value:"7", label:"7 - crocodile(2 classes: African crocodile, American alligator)"},
        {value:"8", label:"8 - dinosaur(1 class: triceratops)"},
        {value:"9", label:"9 - snake(3 classes: thunder snake, ringneck snake, hognose snake)"},
        {value:"-1", label:'...', disable: true}
      ],
      cifar100superclasses: [
            {value:"0", label:"0 - beaver, dolphin, otter, seal, whale"},
            {value:"1", label:"1 - aquarium_fish, flatfish, ray, shark, trout"},
            {value:"2", label:"2 - orchid, poppy, rose, sunflower, tulip"},
            {value:"3", label:"3 - bottle, bowl, can, cup, plate"},
            {value:"4", label:"4 - apple, mushroom, orange, pear, sweet_pepper"},
            {value:"5", label:"5 - clock, keyboard, lamp, telephone, television"},
            {value:"6", label:"6 - bed, chair, couch, table, wardrobe"},
            {value:"7", label:"7 - bee, beetle, butterfly, caterpillar, cockroach"},
            {value:"8", label:"8 - bear, leopard, lion, tiger, wolf"},
            {value:"9", label:"9 - bridge, castle, house, road, skyscraper"},
            {value:"10", label:"10 - cloud, forest, mountain, plain, sea"},
            {value:"11", label:"11 - camel, cattle, chimpanzee, elephant, kangaroo"},
            {value:"12", label:"12 - fox, porcupine, possum, raccoon, skunk"},
            {value:"13", label:"13 - crab, lobster, snail, spider, worm"},
            {value:"14", label:"14 - baby, boy, girl, man, woman"},
            {value:"15", label:"15 - crocodile, dinosaur, lizard, snake, turtle"},
            {value:"16", label:"16 - hamster, mouse, rabbit, shrew, squirrel"},
            {value:"17", label:"17 - maple_tree, oak_tree, palm_tree, pine_tree, willow_tree"},
            {value:"18", label:"18 - bicycle, bus, motorcycle, pickup_truck, train"},
            {value:"19", label:"19 - lawn_mower, rocket, streetcar, tank, tractor"},
      ],
      cifar100classes: [
        {value:"0",label:"0 - beaver"},
        {value:"1",label:"1 - dolphin"},
        {value:"2",label:"2 - otter"},
        {value:"3",label:"3 - seal"},
        {value:"4",label:"4 - whale"},
        {value:"5",label:"5 - aquarium fish"},
        {value:"6",label:"6 - flatfish"},
        {value:"7",label:"7 - ray"},
        {value:"8",label:"8 - shark"},
        {value:"9",label:"9 - trout"},
        {value:"10",label:"10 - orchids"},
        {value:"11",label:"11 - poppies"},
        {value:"12",label:"12 - roses"},
        {value:"13",label:"13 - sunflowers"},
        {value:"14",label:"14 - tulips"},
        {value:"15",label:"15 - bottles"},
        {value:"16",label:"16 - bowls"},
        {value:"17",label:"17 - cans"},
        {value:"18",label:"18 - cups"},
        {value:"19",label:"19 - plates"},
        {value:"20",label:"20 - apples"},
        {value:"21",label:"21 - mushrooms"},
        {value:"22",label:"22 - oranges"},
        {value:"23",label:"23 - pears"},
        {value:"24",label:"24 - sweet peppers"},
        {value:"25",label:"25 - clock"},
        {value:"26",label:"26 - computer keyboard"},
        {value:"27",label:"27 - lamp"},
        {value:"28",label:"28 - telephone"},
        {value:"29",label:"29 - television"},
        {value:"30",label:"30 - bed"},
        {value:"31",label:"31 - chair"},
        {value:"32",label:"32 - couch"},
        {value:"33",label:"33 - table"},
        {value:"34",label:"34 - wardrobe"},
        {value:"35",label:"35 - bee"},
        {value:"36",label:"36 - beetle"},
        {value:"37",label:"37 - butterfly"},
        {value:"38",label:"38 - caterpillar"},
        {value:"39",label:"39 - cockroach"},
        {value:"40",label:"40 - bear"},
        {value:"41",label:"41 - leopard"},
        {value:"42",label:"42 - lion"},
        {value:"43",label:"43 - tiger"},
        {value:"44",label:"44 - wolf"},
        {value:"45",label:"45 - bridge"},
        {value:"46",label:"46 - castle"},
        {value:"47",label:"47 - house"},
        {value:"48",label:"48 - road"},
        {value:"49",label:"49 - skyscraper"},
        {value:"50",label:"50 - cloud"},
        {value:"51",label:"51 - forest"},
        {value:"52",label:"52 - mountain"},
        {value:"53",label:"53 - plain"},
        {value:"54",label:"54 - sea"},
        {value:"55",label:"55 - camel"},
        {value:"56",label:"56 - cattle"},
        {value:"57",label:"57 - chimpanzee"},
        {value:"58",label:"58 - elephant"},
        {value:"59",label:"59 - kangaroo"},
        {value:"60",label:"60 - fox"},
        {value:"61",label:"61 - porcupine"},
        {value:"62",label:"62 - possum"},
        {value:"63",label:"63 - raccoon"},
        {value:"64",label:"64 - skunk"},
        {value:"65",label:"65 - crab"},
        {value:"66",label:"66 - lobster"},
        {value:"67",label:"67 - snail"},
        {value:"68",label:"68 - spider"},
        {value:"69",label:"69 - worm"},
        {value:"70",label:"70 - baby"},
        {value:"71",label:"71 - boy"},
        {value:"72",label:"72 - girl"},
        {value:"73",label:"73 - man"},
        {value:"74",label:"74 - woman"},
        {value:"75",label:"75 - crocodile"},
        {value:"76",label:"76 - dinosaur"},
        {value:"77",label:"77 - lizard"},
        {value:"78",label:"78 - snake"},
        {value:"79",label:"79 - turtle"},
        {value:"80",label:"80 - hamster"},
        {value:"81",label:"81 - mouse"},
        {value:"82",label:"82 - rabbit"},
        {value:"83",label:"83 - shrew"},
        {value:"84",label:"84 - squirrel"},
        {value:"85",label:"85 - maple"},
        {value:"86",label:"86 - oak"},
        {value:"87",label:"87 - palm"},
        {value:"88",label:"88 - pine"},
        {value:"89",label:"89 - willow"},
        {value:"90",label:"90 - bicycle"},
        {value:"91",label:"91 - bus"},
        {value:"92",label:"92 - motorcycle"},
        {value:"93",label:"93 - pickup truck"},
        {value:"94",label:"94 - train"},
        {value:"95",label:"95 - lawn-mower"},
        {value:"96",label:"96 - rocket"},
        {value:"97",label:"97 - streetcar"},
        {value:"98",label:"98 - tank"},
        {value:"99",label:"99 - tractor"}
    ],
      
      vggsList:[
        // {value:'vgg11', label:'VGG11', disabled:false},
        {value:'vgg11_bn', label:'VGG11_bn', disabled:false},
        // {value:'vgg13', label:'VGG13', disabled:false},
        {value:'vgg13_bn', label:'VGG13_bn', disabled:false},
        // {value:'vgg16', label:'VGG16', disabled:false},
        {value:'vgg16_bn', label:'VGG16_bn', disabled:false},
        // {value:'vgg19', label:'VGG19', disabled:false},
        {value:'vgg19_bn', label:'VGG19_bn', disabled:false},
      ],
      resnetsList:[
        //'resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152'
        {value:'resnet18', label:'ResNet18', disabled:false},
        {value:'resnet34', label:'ResNet34', disabled:false},
        {value:'resnet50', label:'ResNet50', disabled:false},
        {value:'resnet101', label:'ResNet101', disabled:false},
        {value:'resnet152', label:'ResNet152', disabled:false},
      ]
  
    };
  },
  
  
  computed: {
    wholeDatasetFileOptions(){
      switch(this.modelFile){
          case 'vgg16':
            return[{value:'cifar10', label:'CIFAR-10'}, {value:'cifar100', label:'CIFAR-100'}]
          case 'resnet20':
            return[{value:'cifar10', label:'CIFAR-10'}, {value:'cifar100', label:'CIFAR-100'}]
          case 'resnet50':
            return [{value:'ImageNet', label:'ImageNet'}]
          case 'resnet18':
            return [{value:'mit67', label:'MIT67'}] 
          default:
            return [{value:'cifar10', label:'CIFAR-10'}, {value:'svhn', label:'SVHN'}] 
      }
    },
    wholeModelFileOptions(){
      if(this.fromSelection === 'true'){
        switch(this.datasetFile){
          case 'cifar10':
            return [
            {label:'VGGs', options:this.vggsList, },
            {label:'Other Models',
              options:[ 
              {value:'resnet20', label:'ResNet20', disabled:false},
              {value:'simcnn', label:'SimCNN', disabled:false}, 
              {value:'rescnn', label:'ResCNN', disabled:false}, 
              {value:'incecnn', label:'InceCNN', disabled:false}
              ]
            }]
          case 'cifar100':
            return[
              {label:'VGGs', options:this.vggsList, },
              {label:'ResNets', options:this.resnetsList,},
              {label:'Other Models',
              options:[ 
                {value:'resnet20', label:'ResNet20', disabled:false},
              ]
            }]
          case 'imagenet':
            return[
              {label:'VGGs', options:this.vggsList, },
              {label:'ResNets', options:this.resnetsList,},
            ]
          case 'svhn':
            return[
              {label:'VGGs', options:this.vggsList, },
              {label:'Other Models',
                options:[ 
                  {value:'simcnn', label:'SimCNN', disabled:false}, 
                  {value:'rescnn', label:'ResCNN', disabled:false}, 
                  {value:'incecnn', label:'InceCNN', disabled:false}
                ]
            }]
        }
      }
      else{
        return [
        {label:'VGGs', options:this.vggsList, },
        {label:'ResNets', options:this.resnetsList,},
        {
          label:'Other Models',
          options:[
            {value:'resnet20', label:'ResNet20', disabled:false},
            {value:'simcnn', label:'SimCNN', disabled:false}, 
            {value:'rescnn', label:'ResCNN', disabled:false}, 
            {value:'incecnn', label:'InceCNN', disabled:false}
          ]
        }] 
      }
    }
  },
  methods: {
    // router Methods
    JumpToDeployment(){ this.$router.push('/deployment') },
    JumpToBenchmark(){ this.$router.push('/benchmark') },
    
    // reset method
    ResetModelandDataset(){
      if(this.datasetFileUploadMode === '1' ){this.datasetFile = null;};
      if(this.modelFileUploadMode === '1'){this.modelFile = null}
    },
    ResetAll(){
      if(this.fromSelection != 'true'){
        this.datasetFile = null
      }
      this.algorithm = ''
      this.selectDisabled = {SEAM: false, GradSplitter: false, binary:false, multi:false, defect:false,}
      this.directModelReuse = ''
      this.learningRate = 0.01
      this.alpha = 1
      
    },
  
    // 根据选择的模型，数据集和reuse method 修改target class和target super class的选项
    settargetIdxOptions(){
      let that = this 
      if(this.directModelReuse === 'Multi-Class Classification'){
        if(this.datasetFile === 'ImageNet'){
          console.log('set targetSUPERclass for ImageNet')
          this.targetSuperclassIdxOptions = that.imagenetsuperclasses
        }
        else if (this.datasetFile === 'cifar100'){
          console.log('set targetSUPERclass for CIFAR-100')
          this.targetSuperclassIdxOptions = this.cifar100superclasses
          if (this.modelFile == 'resnet20'){
            this.alpha = 2
          }
        }
      }
      if(this.directModelReuse === 'Binary Classification'){
        if(this.datasetFile === 'cifar10'){
          console.log('set targetclass for cifar10')
          this.targetClassIdxOptions= this.cifar10classes
        }
        else if(this.datasetFile === 'cifar100'){
          console.log('set targetclass for cifar100')
          this.targetClassIdxOptions = this.cifar100classes
          if (this.modelFile == 'resnet20'){
            this.alpha = 1.5
          }
        }
      }
    },
  
    //控制是否弹出确认dialog
    beforeRunModularization(){
      if(this.algorithm === 'GradSplitter'){
        console.log('before run gradsplitter')
        this.AreUSure = true
      }
      else{
        console.log('before run seam')
        let that = this
        that.runModularization()
      }
    },
  
    //前端主要逻辑，辅助修改各项参数
    AlgorithSelectHelper(){
      /* *
      * input: model, dataset
      * output: this.algorithm
      * ATTENTION: this.$set(this.algorithm, xxx)
      * */
     let that = this
     // VGG系列(8个模型)，算法支持SeaM(Bi)和Grad，数据集支持cifar10/cifar100/svhn
     if(this.modelFile.startsWith('vgg') && this.datasetFile!=null){
    //  if(this.modelFile === 'vgg16' && this.datasetFile!=null){
      console.log(this.modelFile)
      console.log(this.datasetFile)
        if(this.datasetFile === 'cifar100' || this.datasetFile ==='ImageNet'){
          this.algorithm= 'SEAM'
          this.selectDisabled.GradSplitter = true
        }
        this.selectDisabled.multi = true
        this.selectDisabled.defect = true
        this.directModelReuse = 'Binary Classification'
        that.settargetIdxOptions()
        this.learningRate = 0.01
        this.alpha = 1
        if(this.datasetFile === 'cifar100'){
          this.learningRate = 0.05
        }
     }
     //resnet 20(不属于Resnets)，支持SeaM的cifar-10和cifar-100
     if (this.modelFile === 'resnet20'){
        if(this.datasetFile === 'cifar10'){
          this.algorithm= 'SEAM'
          this.selectDisabled.GradSplitter = true
          this.selectDisabled.multi = true
          this.selectDisabled.defect = true
          this.directModelReuse = 'Binary Classification'
          that.settargetIdxOptions()
          this.learningRate = 0.05
          this.alpha = 1
        }
        else if(this.datasetFile === 'cifar100'){
          this.algorithm= 'SEAM'
          this.directModelReuse = ''
          this.selectDisabled.GradSplitter = true
          this.selectDisabled.defect = true 
          this.selectDisabled.multi = false
          this.selectDisabled.binary = false
          this.learningRate = 0.1
          this.alpha = 1.5
        }
     }
    //  if(this.modelFile === 'resnet50' && this.datasetFile === 'ImageNet'){
    //     this.algorithm= 'SEAM'
    //     this.selectDisabled.GradSplitter = true
    //     this.selectDisabled.binary = true
    //     this.selectDisabled.defect = true
    //     this.directModelReuse = 'Multi-Class Classification'
    //     that.settargetIdxOptions()
    //     this.learningRate = 0.05
    //     this.alpha = 2
    //  }
    //ResNets+ImageNet+Seam(Multi)
     if((this.datasetFile === 'ImageNet')|| (this.modelFile.startsWith('resnet') && this.modelFile != 'resnet20')){
        this.algorithm= 'SEAM'
        this.selectDisabled.GradSplitter = true
        this.selectDisabled.binary = true
        this.selectDisabled.defect = true
        this.directModelReuse = 'Multi-Class Classification'
        that.settargetIdxOptions()
        this.learningRate = 0.1
        this.alpha = 2
     }
     if(this.modelFile === 'resnet18' && this.datasetFile === 'mit67'){ 
        this.algorithm= 'SEAM'
        this.selectDisabled.GradSplitter = true
        this.selectDisabled.binary = true
        this.selectDisabled.multi = true
        this.directModelReuse = 'Defect Inheritance'
        this.learningRate = 0.05
        this.alpha = 0.5
     }
     if((this.modelFile === 'simcnn' || this.modelFile === 'rescnn' || this.modelFile === 'incecnn') && this.datasetFile != null){
        // this.algorithm ='GradSplitter'
        this.selectDisabled.SEAM = false
        this.selectDisabled.multi = true
        this.selectDisabled.binary = false
        this.directModelReuse = 'Binary Classification'
        // seam后来加的模型，所有binary，lr=0.01,alpha=1；muiti：lr=0.1,alpha=2.0
        this.learningRate = 0.01
        this.alpha = 1
        // this.reuseMethod = ''
    }
    },

    AlgorithSelectHelperFromSelection(fromSelection){
      console.log(this.modelFile)
      /* *
      * input: model, dataset
      * output: this.algorithm
      * ATTENTION: this.$set(this.algorithm, xxx)
      * [dataset can not be changed]
      * */
      if(fromSelection!='true'){
        console.log('from selection != true');
        return
      }

      let that = this
      that.ResetAll()
      console.log(this.datasetFile)
      that.AlgorithSelectHelper()
    },
  
    // 与后段交互相关函数
    // 轮询函数，更新右侧栏中的排队部分
    updateProgress(){
      axios.get('http://127.0.0.1:5000/list_tasks')
        .then(response => {
          //{task_id:status, task_id:status, ...}
          console.log(response.data)
          let taskQueue = response.data
          console.log(taskQueue)  
          this.taskQueue = taskQueue
  
          //get queueNum
          const statuses = Object.values(taskQueue)
          console.log(statuses)
          var count_p=0, count_r=0, count_d=0
          if (statuses.length>0){
            count_p = statuses.filter(item => item== 'pending').length;
            count_r = statuses.filter(item => item== 'running').length;
            count_d = statuses.filter(item => item== 'done').length;
          }
          this.queueNum.pending = count_p
          this.queueNum.running = count_r
          this.queueNum.done = count_d
          console.log(JSON.stringify(this.queueNum))  // queueNum:{pending:0,running: 0, done:0,},
          //get taskStatus 
          if(this.taskid){
            const descriptor = Object.getOwnPropertyDescriptor(taskQueue, this.taskid);
            console.log(descriptor.value)
            this.$set(this.taskStatus, descriptor.value)
          }
          //get lenQueue
          const len = Object.keys(taskQueue).length
          this.lenQueue = len
          this.$forceUpdate()
        })
        .catch(error => {
          console.error(error);
        });
    },
    // 运行modularization 
    runModularization(){
      console.log('run modular')
      if(this.AreUSure === true){
        this.AreUSure = false
      }
      this.$message({message:'RUNNING...',  type: 'success'}, {center:true, showConfirmButton:false})
      const data = {
        modelFile: this.modelFile,
        datasetFile: this.datasetFile,
        algorithm: (this.algorithm).toString(),
        epoch: (this.epoch).toString(),
        learningRate: (this.learningRate).toString(),
        directModelReuse: this.directModelReuse,
        targetClass: this.targetClass.value,
        targetClassLabel: this.targetClass.label,
        alpha: (this.alpha).toString(),
        targetSuperclassIdx: this.targetSuperclassIdx.value,
        targetSuperclassLabel: this.targetSuperclassIdx.label,
      };
      console.log(data)
      sessionStorage.setItem("modelFile", this.modelFile);
      sessionStorage.setItem("datasetFile", this.datasetFile);
      sessionStorage.setItem("algorithm", this.algorithm);
      sessionStorage.setItem("epoch", this.epoch);
      sessionStorage.setItem("learningRate", this.learningRate);
      sessionStorage.setItem("directModelReuse", this.directModelReuse);
      sessionStorage.setItem("targetClass", this.targetClass.value);
      sessionStorage.setItem("targetClassLabel", this.targetClass.label);
      sessionStorage.setItem("alpha", this.alpha);
      sessionStorage.setItem("targetSuperclassIdx", this.targetSuperclassIdx.value);
      sessionStorage.setItem("targetSuperclassLabel", this.targetSuperclassIdx.label);
  
      this.progressrunning = true
      // Send POST requests to Flask
      axios.post('http://127.0.0.1:5000/run_model', data)
        .then(response => {
          // success, return results
          // this.logs = response.data.logs;
          // this.isModelReady = response.data.isModelReady;
          this.taskid = response.data.task_id
          console.log('got task_id =' + this.taskid)
        })
        .catch(error => {
          // return errors
          console.error(error);
          // this.logs = 'An error occurred while running the model.';
        });
  
        sessionStorage.setItem("taskid", this.taskid)
    },
    // 运行reuse
    runReuse(){
      this.$message({message:'RUNNING...',  type: 'success'}, {center:true, showConfirmButton:false})
      const data = {
        modelFile: sessionStorage.getItem("modelFile"),
        datasetFile: sessionStorage.getItem("datasetFile"),
        algorithm: (this.algorithm).toString(),
        epoch: (this.epoch).toString(),
        reuseMethod: this.reuseMethod,
        cifarclass: this.cifarclass,
        svhnclass: this.svhnclass,
      };
      console.log(data)
      // Send POST requests to Flask
      axios.post('http://127.0.0.1:5000/run_reuse', data)
        .then(response => {
        })
        .catch(error => {
          console.error(error);
          this.reuselogs = 'An error occurred while running the model.';
        });
    },
    // 下载模块
    download() {
      const data = {
        modelFile: this.modelFile,
        datasetFile: this.datasetFile,
        algorithm: (this.algorithm).toString(),
        epoch: (this.epoch).toString(),
        learningRate: (this.learningRate).toString(),
        directModelReuse: this.directModelReuse,
        targetClass: this.targetClass.value,
        alpha: (this.alpha).toString(),
        targetSuperclassIdx: this.targetSuperclassIdx.value,
      };
      axios.post('http://127.0.0.1:5000/download', data, {
        responseType: 'blob'
      })
      .then(response => {
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        // 从响应头中拿文件名
        const fileName = response.headers['content-disposition'];
        console.log('downloadName: ' + fileName);
        console.log(response.headers);
        link.setAttribute('download', fileName);
        document.body.appendChild(link);
        link.click();
      })
      .catch(error => {
        console.error("Download error: ", error);
      });
  
    },  
    checkRunInformation(){
      console.log('form:'+this.form)
      console.log('algorithm:'+this.algorithm)
      console.log('directModelReuse:'+this.directModelReuse)
      console.log('targetSuperclassIdx:'+this.targetSuperclassIdx)
      console.log('targetClass:'+this.targetClass)
      console.log('epoch:'+this.epoch)
      console.log('alpha:'+this.alpha)
      console.log('learningRate:'+this.learningRate)
      console.log('modelFile:'+this.modelFile)
      console.log('datasetFile:'+this.datasetFile)
    },
  
  },
  };
  
  </script>
  
  <style>
  .reuseclass{
  font-family: Arial, sans-serif;
  color: #333;
  
  }
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
  .el-aside{
    padding:20px;
    color:#333;
  }
  .el-tag--dark.is-hit {
      border:0;
      border-radius: 30px;
  }
  .el-form-item {
    margin-right: 100px;
  }
  .form-body {
    margin-top: 60px;
  }
  .selectLong .el-input{
    width:450px;
  }
  .selectItem .el-form-item__label{
    font-size: large;
    font-weight: bold;
  }
  .selectItemMini .el-form-item__label{
    font-weight: bold;
    font-size: medium;
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
  .areusuredialog .el-dialog__body{
    padding: 10px 30px;
    word-break: normal;
  }
  .areusuredialog .el-dialog__header{
    padding-bottom: 0;
    font-family: Avenir, Helvetica, Arial, sans-serif;
  }
  </style>