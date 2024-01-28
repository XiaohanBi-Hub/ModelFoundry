<template>
    <el-container>
      <!-- Header -->
      <el-header style="height: 85px;"> <span style="margin-top: 0;margin-left: 5%;"> 
        ModelFoundry
      </span> </el-header>
  
        <!-- Main body -->
        <el-main>
          <!-- Title -->
          <h2 style="text-align: left;margin-left: 15%;float: left;margin-top: 2%;margin-bottom: 3%;">
                Search Modules</h2>
          
          <el-button style="float:right;margin-right: 18%;margin-top: 2%;font-size: larger;width:220px;margin-bottom: 10px;" 
                      type="success" @click="JumpToModularizationDirectly">
                    Modularization</el-button>
          
          <div class="main-body" style="margin-left: 20%;margin-right: 5%;margin-top: 100px;">
            <div class="search" style="margin-bottom: 50px;text-align:left;">
              <el-row style="width:85%">
                    <el-col :span="24">
                      <p style="font-size: larger;font-weight: bold;margin-bottom: 20px;">Module Selection</p>
                    </el-col>
                    <el-col :span="24">
                      <el-select v-model="moduleNameInput" filterable placeholder="Choose a module" style="width: 97%;" @change="seachModuleByName(moduleNameInput)">
                        <el-option
                          v-for="item in ModuleList"
                          :key="item.value"
                          :label="item.value"
                          :value="item.value">
                        </el-option>
                      </el-select>  
                    </el-col>
                </el-row>
                
            </div>
            <div style="width:82%"><el-divider></el-divider></div>
            
            <div class="result">
                <!-- <el-row style="width:85%">
                    <el-col :span="8" >
                    </el-col>
                </el-row> -->
                <div class="card-container" style="display: flex;flex-wrap: wrap;">
                  <div v-for="item in ClassResultList" style="width: 20%;margin-bottom: 15px;min-width: 200px;margin-right: 10px;">
                  <el-card shadow="hover" @click.native="JumpToModularization(item)" style="height: 180px;">
                      <div slot="header" class="clearfix">
                          <span style="font-size: large;font-weight: bold;margin-right: 15px;"> {{item.module_name}}</span>
                          <span><el-tag v-if="item.tag == 'classes' " type="success" style="">class</el-tag>
                              <el-tag v-else type="warning"  style="margin-left: 15px;">superclass</el-tag>
                          </span>  
                      </div>
                      <div style="font-style: italic;word-break: break-word;">from: {{item.dataset}}</div>
                  </el-card></div>
                </div>
            </div>
          </div>
        </el-main>
        

      </el-container>
  
  </template>
  
  <script>
  import axios from 'axios';
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
  },
  data() {
    return {
      ClassResultList:[
        { module_name:'cat',
          dataset:'cifar10_classes',
          tag:'super classes',
          idx: '1'
        },
        {
          module_name:'dog',
          dataset:'cifar10:cute dogs',
          tag:'classes'

        },
        {
          module_name:'number 1',
          dataset:'SVHN',
          tag:'classes'

        },
        {
          module_name:'number 1',
          dataset:'SVHN',
          tag:'classes'

        },
        {
          module_name:'number 1',
          dataset:'SVHN',
          tag:'classes'

        }
      ],
      ModuleList:[{value: "accessory"}, {value: "aircraft"}, {value: "airplane"}, {value: "apples"}, {value: "aquarium fish"}, {value: "aquatic mammals"}, {value: "arachnid"}, {value: "armadillo"}, {value: "automobile"}, {value: "baby"}, {value: "ball"}, {value: "bear"}, {value: "beaver"}, {value: "bed"}, {value: "bee"}, {value: "beetle"}, {value: "bicycle"}, {value: "bird"}, {value: "boat"}, {value: "bottles"}, {value: "bowls"}, {value: "boy"}, {value: "bridge"}, {value: "bug"}, {value: "building"}, {value: "bus"}, {value: "butterfly"}, {value: "camel"}, {value: "cans"}, {value: "castle"}, {value: "cat"}, {value: "caterpillar"}, {value: "cattle"}, {value: "chair"}, {value: "chimpanzee"}, {value: "clock"}, {value: "clothing"}, {value: "cloud"}, {value: "cockroach"}, {value: "computer keyboard"}, {value: "container"}, {value: "cooking"}, {value: "coral"}, {value: "couch"}, {value: "crab"}, {value: "crocodile"}, {value: "crustacean"}, {value: "cups"}, {value: "decor"}, {value: "deer"}, {value: "dinosaur"}, {value: "dog"}, {value: "dolphin"}, {value: "echinoderms"}, {value: "electronics"}, {value: "elephant"}, {value: "fence"}, {value: "ferret"}, {value: "fish"}, {value: "flatfish"}, {value: "flower"}, {value: "flowers"}, {value: "food"}, {value: "food containers"}, {value: "forest"}, {value: "fox"}, {value: "frog"}, {value: "fruit"}, {value: "fruit and vegetables"}, {value: "fungus"}, {value: "furniture"}, {value: "girl"}, {value: "hamster"}, {value: "hat"}, {value: "hog"}, {value: "horse"}, {value: "house"}, {value: "household electrical devices"}, {value: "household furniture"}, {value: "insects"}, {value: "instrument"}, {value: "kangaroo"}, {value: "lab equipment"}, {value: "lamp"}, {value: "large carnivores"}, {value: "large man-made outdoor things"}, {value: "large natural outdoor scenes"}, {value: "large omnivores and herbivores"}, {value: "lawn-mower"}, {value: "leopard"}, {value: "lion"}, {value: "lizard"}, {value: "lobster"}, {value: "man"}, {value: "maple"}, {value: "marine mammals"}, {value: "marsupial"}, {value: "medium-sized mammals"}, {value: "mollusk"}, {value: "mongoose"}, {value: "monotreme"}, {value: "motorcycle"}, {value: "mountain"}, {value: "mouse"}, {value: "mushrooms"}, {value: "non-insect invertebrates"}, {value: "oak"}, {value: "oranges"}, {value: "orchids"}, {value: "other"}, {value: "otter"}, {value: "outdoor scene"}, {value: "palm"}, {value: "paper"}, {value: "pears"}, {value: "people"}, {value: "person"}, {value: "pickup truck"}, {value: "pine"}, {value: "plain"}, {value: "plant"}, {value: "plates"}, {value: "poppies"}, {value: "porcupine"}, {value: "possum"}, {value: "primate"}, {value: "rabbit"}, {value: "raccoon"}, {value: "ray"}, {value: "reptiles"}, {value: "road"}, {value: "rocket"}, {value: "rodent"}, {value: "roses"}, {value: "salamander"}, {value: "sea"}, {value: "seal"}, {value: "shark"}, {value: "ship"}, {value: "shrew"}, {value: "skunk"}, {value: "skyscraper"}, {value: "sloth"}, {value: "small mammals"}, {value: "snail"}, {value: "snake"}, {value: "spider"}, {value: "sports equipment"}, {value: "squirrel"}, {value: "streetcar"}, {value: "sunflowers"}, {value: "sweet peppers"}, {value: "table"}, {value: "tank"}, {value: "technology"}, {value: "telephone"}, {value: "television"}, {value: "tiger"}, {value: "tool"}, {value: "toy"}, {value: "tractor"}, {value: "train"}, {value: "trees"}, {value: "trilobite"}, {value: "trout"}, {value: "truck"}, {value: "tulips"}, {value: "turtle"}, {value: "ungulate"}, {value: "vegetable"}, {value: "vehicle"}, {value: "vehicles 1"}, {value: "vehicles 2"}, {value: "wardrobe"}, {value: "weapon"}, {value: "whale"}, {value: "wild cat"}, {value: "wild dog"}, {value: "willow"}, {value: "wolf"}, {value: "woman"}, {value: "worm"},{value: '0'},{value: '1'},{value: '2'},{value: '3'},{value: '4'},{value: '5'},{value: '6'},{value: '7'},{value: '8'},{value: '9'},],
      moduleNameInput:'',
  
    };
  },
  
  methods: {
    // router Methods
    JumpToDeployment(){ this.$router.push('/deployment') },
    JumpToModularizationDirectly(){
      sessionStorage.setItem("fromSelection", false);
      this.$router.push('/modularization') 
    },
    JumpToModularization(item){ 
      // item.module_name/tag(=classes/superclasses)/dataset(=cifar10_classes,cifar100_superclasses)
      sessionStorage.setItem("fromSelection", true);
      var ds_l=item.dataset.split("_");
      var target_ds = '';
      console.log(ds_l[0])
      if(ds_l[0] === 'cifar10'||ds_l[0] === 'cifar100'||ds_l[0] === 'svhn'){ target_ds=ds_l[0]; }
      else if(ds_l[0] === 'imagenet'){ target_ds='ImageNet';  }
      else {target_ds = 'somethingWrong'}
      //dataset: cifar10/cifar100/ImageNet/svhn
      sessionStorage.setItem("datasetFile", target_ds);

      var module_tag = ds_l[1]
      sessionStorage.setItem("module_tag", module_tag)
      sessionStorage.setItem("module_idx", item.idx)
      sessionStorage.setItem("module_name",item.module_name)

      // sessionStorage.setItem("directModelReuse", this.directModelReuse);
      // sessionStorage.setItem("targetClass", this.targetClass.value);
      // sessionStorage.setItem("targetClassLabel", this.targetClass.label);
      // sessionStorage.setItem("alpha", this.alpha);
      // sessionStorage.setItem("targetSuperclassIdx", this.targetSuperclassIdx.value);
      // sessionStorage.setItem("targetSuperclassLabel", this.targetSuperclassIdx.label);
       
      this.$router.push('/modularization') 
    },

    // 与后段交互相关函数
    seachModuleByName(moduleNameInput){
      const data = {
        searchClass: this.moduleNameInput,
      };
      console.log(data)
      axios.post('http://127.0.0.1:5000/search_module', data)
        .then(response => {
          console.log(response.data)
          this.ClassResultList = response.data
        })
        .catch(error => {
          // return errors
          console.error(error);
          this.logs = 'An error occurred';
        });
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
    color: #333;
    text-align: left;
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