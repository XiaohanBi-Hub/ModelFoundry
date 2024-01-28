import Vue from "vue"
import VueRouter from "vue-router"

import MyComponent from '../components/MyComponent.vue'
import Modularization from '../components/Modularization.vue'
import test from '../components/Modularization_old.vue'
import Deployment from '../components/Deployment.vue'
import Benchmark from '../components/Benchmark'
import Search from '../components/SearchModules'

Vue.use(VueRouter)


const router = new VueRouter({
    routes:[
        { path: '/', redirect:'/searchmodules' },
        { path:'/searchmodules', component: Search, meta:{title: 'ModReuser'}},
        { path:'/modularization', component: Modularization, meta:{title: 'ModReuser Modularization'}},
        { path:'/deployment', component: Deployment, meta:{title: 'ModReuser Deployment'}},
        { path:'/benchmark', component: Benchmark , meta:{title: 'ModReuser Benchmark'}},
        { path:'/testtestestsetsetset', component: test },

        { path:'/oldbxh', component: MyComponent , meta:{title: 'Oh No DONT LOOK AT MEEEEE'}},
    ]
})


router.beforeEach((to, from, next) => {
    if (to.meta.title) {
      document.title = to.meta.title
    }
    next()
  })


export default router

