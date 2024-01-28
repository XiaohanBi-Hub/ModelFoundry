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
                  <el-select v-model="modelFile" placeholder="Please select a model" @change="ResetAll">
                      <el-option  v-for="item in ModelFileList" 
                        :key="item.value" :label="item.label" :value="item.value">
                        </el-option>
                    </el-select>
                </el-form-item>

                <!-- Dataset Selection -->
                <el-form-item label="Dataset File:" class="selectItem" style="width: 100%;">
                  <el-select v-model="datasetFile" placeholder="Please select a dataset" @change="AlgorithSelectHelper">
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
                          <el-card style="width: 90%;height: 120px;" :body-style="{ padding: '0px 20px' }"  
                          :shadow="(this.directModelReuse === 'Defect Inheritance')? 'always' : 'hover'">
                            <p style="font-weight: bolder;margin-top: 0;margin-bottom:15px;color:#606266">Indirect Model Reuse</p>
                            <el-radio-group v-model="directModelReuse">
                              <el-radio label="Defect Inheritance" size="medium" style="margin-bottom: 15px;" :disabled="selectDisabled.defect">Defect Inheritance</el-radio>
                            </el-radio-group>
                          </el-card>
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
                              <el-radio label="For New Task"  size="medium" style="margin-bottom: 15px;" >For New Task</el-radio>
                            </el-radio-group>
                          </el-card>
                        </div></el-col></el-row>
                    </div>
                </el-form-item>
                <el-form-item v-if="algorithm === 'GradSplitter' && reuseMethod==='For New Task'" label="Choose Modules:" class="selectItem">
                  <el-select v-model="cifarclass" placeholder="Modules from CIFAR-10" style="margin-bottom: 10px;margin-right: 20px;">
                    <el-option v-for="item in cifar10classes" :key="item.value" :label="item.label" :value="item.value"></el-option>
                  </el-select> 
                  <el-select v-model="svhnclass" placeholder="Modules from SVHN" >
                    <el-option v-for="item in svhnclasses" :key="item.value" :label="item.label" :value="item.value"></el-option>
                  </el-select>
                  <p style="color: darkgray;margin-top: 0;line-height: 10px;">Build a binary-classification model by combination two modules. </p>
                </el-form-item>

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
                    <el-select v-model="targetClass" 
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
              Download Process Model</el-button>
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
  this.datasetFile = sessionStorage.getItem("datasetFile")

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
    fromSelection: false, //passed from selection

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
    targetClass: '',  // The target class selected by the user
    alpha: 1,  // The alpha value entered by the user, default to 1
    targetSuperclassIdx: '',  // The target superclass index selected by the user
    
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
    
    reuseMethod:'', // in Model Reuse [=More Accurate/For New Task]
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
      {value:'vgg16', label:'VGG16'},
      {value:'resnet20', label:'ResNet20'},
      {value:'resnet50', label:'ResNet50'},
      {value:'resnet18', label:'ResNet18'},
      {value:'simcnn', label:'SimCNN'}, 
      {value:'rescnn', label:'ResCNN'}, 
      {value:'incecnn', label:'InceCNN'}
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
    cifar100classes:[
      {value:"0",label:"0 - apple"},
      {value:"1",label:"1 - aquarium_fish"},
      {value:"2",label:"2 - baby"},
      {value:"3",label:"3 - bear"},
      {value:"4",label:"4 - beaver"},
      {value:"5",label:"5 - bed"},
      {value:"6",label:"6 - bee"},
      {value:"7",label:"7 - beetle"},
      {value:"8",label:"8 - bicycle"},
      {value:"9",label:"9 - bottle"},
      {value:"10",label:"10 - bowl"},
      {value:"11",label:"11 - boy"},
      {value:"12",label:"12 - bridge"},
      {value:"13",label:"13 - bus"},
      {value:"14",label:"14 - butterfly"},
      {value:"15",label:"15 - camel"},
      {value:"16",label:"16 - can"},
      {value:"17",label:"17 - castle"},
      {value:"18",label:"18 - caterpillar"},
      {value:"19",label:"19 - cattle"},
      {value:"20",label:"20 - chair"},
      {value:"21",label:"21 - chimpanzee"},
      {value:"22",label:"22 - clock"},
      {value:"23",label:"23 - cloud"},
      {value:"24",label:"24 - cockroach"},
      {value:"25",label:"25 - couch"},
      {value:"26",label:"26 - crab"},
      {value:"27",label:"27 - crocodile"},
      {value:"28",label:"28 - cup"},
      {value:"29",label:"29 - dinosaur"},
      {value:"30",label:"30 - dolphin"},
      {value:"31",label:"31 - elephant"},
      {value:"32",label:"32 - flatfish"},
      {value:"33",label:"33 - forest"},
      {value:"34",label:"34 - fox"},
      {value:"35",label:"35 - girl"},
      {value:"36",label:"36 - hamster"},
      {value:"37",label:"37 - house"},
      {value:"38",label:"38 - kangaroo"},
      {value:"39",label:"39 - keyboard"},
      {value:"40",label:"40 - lamp"},
      {value:"41",label:"41 - lawn_mower"},
      {value:"42",label:"42 - leopard"},
      {value:"43",label:"43 - lion"},
      {value:"44",label:"44 - lizard"},
      {value:"45",label:"45 - lobster"},
      {value:"46",label:"46 - man"},
      {value:"47",label:"47 - maple_tree"},
      {value:"48",label:"48 - motorcycle"},
      {value:"49",label:"49 - mountain"},
      {value:"50",label:"50 - mouse"},
      {value:"51",label:"51 - mushroom"},
      {value:"52",label:"52 - oak_tree"},
      {value:"53",label:"53 - orange"},
      {value:"54",label:"54 - orchid"},
      {value:"55",label:"55 - otter"},
      {value:"56",label:"56 - palm_tree"},
      {value:"57",label:"57 - pear"},
      {value:"58",label:"58 - pickup_truck"},
      {value:"59",label:"59 - pine_tree"},
      {value:"60",label:"60 - plain"},
      {value:"61",label:"61 - plate"},
      {value:"62",label:"62 - poppy"},
      {value:"63",label:"63 - porcupine"},
      {value:"64",label:"64 - possum"},
      {value:"65",label:"65 - rabbit"},
      {value:"66",label:"66 - raccoon"},
      {value:"67",label:"67 - ray"},
      {value:"68",label:"68 - road"},
      {value:"69",label:"69 - rocket"},
      {value:"70",label:"70 - rose"},
      {value:"71",label:"71 - sea"},
      {value:"72",label:"72 - seal"},
      {value:"73",label:"73 - shark"},
      {value:"74",label:"74 - shrew"},
      {value:"75",label:"75 - skunk"},
      {value:"76",label:"76 - skyscraper"},
      {value:"77",label:"77 - snail"},
      {value:"78",label:"78 - snake"},
      {value:"79",label:"79 - spider"},
      {value:"80",label:"80 - squirrel"},
      {value:"81",label:"81 - streetcar"},
      {value:"82",label:"82 - sunflower"},
      {value:"83",label:"83 - sweet_pepper"},
      {value:"84",label:"84 - table"},
      {value:"85",label:"85 - tank"},
      {value:"86",label:"86 - telephone"},
      {value:"87",label:"87 - television"},
      {value:"88",label:"88 - tiger"},
      {value:"89",label:"89 - tractor"},
      {value:"90",label:"90 - train"},
      {value:"91",label:"91 - trout"},
      {value:"92",label:"92 - tulip"},
      {value:"93",label:"93 - turtle"},
      {value:"94",label:"94 - wardrobe"},
      {value:"95",label:"95 - whale"},
      {value:"96",label:"96 - willow_tree"},
      {value:"97",label:"97 - wolf"},
      {value:"98",label:"98 - woman"},
      {value:"99",label:"99 - worm"},
    ],
    

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
    this.datasetFile = null
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
   if(this.modelFile === 'vgg16' && this.datasetFile!=null){
    console.log(this.modelFile)
    console.log(this.datasetFile)
      this.algorithm= 'SEAM'
      this.selectDisabled.GradSplitter = true
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
   if(this.modelFile === 'resnet50' && this.datasetFile === 'ImageNet'){
      this.algorithm= 'SEAM'
      this.selectDisabled.GradSplitter = true
      this.selectDisabled.binary = true
      this.selectDisabled.defect = true
      this.directModelReuse = 'Multi-Class Classification'
      that.settargetIdxOptions()
      this.learningRate = 0.05
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
      this.algorithm ='GradSplitter'
      this.selectDisabled.SEAM = true
      this.reuseMethod = ''
  }
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
        this.logs = 'An error occurred while running the model.';
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
      targetClass: this.targetClass,
      alpha: (this.alpha).toString(),
      targetSuperclassIdx: this.targetSuperclassIdx,
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