
<template>
  <div id="app">
    <router-view></router-view>
  </div>
</template>

<script>
// import MyComponent from './components/MyComponent'

export default {
  name: 'App', title:"ModelFoundry"

}
</script>

<style>
html,
body {
  margin: 0px;
  height: 100%;
}
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  /* text-align: center; */
  color: #2c3e50;
  height: 100%;
  /* position:top;
  top:0;bottom: 0;right: 0;left:0; */
  /* margin-top: 60px; */
  scrollbar-width: 0;
}


#app ::-webkit-scrollbar {
  width: 0!important ; display: none;
}
::-webkit-scrollbar {
  width: 0 !important;
  height: 0;
}
</style>