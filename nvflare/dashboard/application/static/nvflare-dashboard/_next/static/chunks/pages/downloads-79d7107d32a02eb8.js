(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[526],{14682:function(e,t,n){"use strict";n.d(t,{y:function(){return g}});var r=n(9669),o=n.n(r),i=n(84506),s=n(40203),a=n(35823),l=n.n(a),c=o().create({baseURL:i.url_root+"/api/v1/",headers:{"Access-Control-Allow-Origin":"*"}});c.interceptors.request.use(function(e){return e.headers.Authorization="Bearer "+(0,s.Gg)().user.token,e},function(e){console.log("Interceptor request error: "+e)}),c.interceptors.response.use(function(e){return e},function(e){throw console.log(" AXIOS error: "),console.log(e),401===e.response.status&&(0,s.KY)("","",-1,0,"",!0),403===e.response.status&&console.log("Error: "+e.response.data.status),404===e.response.status&&console.log("Error: "+e.response.data.status),409===e.response.status&&console.log("Error: "+e.response.data.status),422===e.response.status&&(0,s.KY)("","",-1,0,"",!0),e});var u=function(e){return e.data},d=function(e){return c.get(e).then(u)},p=function(e,t,n){return c.post(e,{pin:n},{responseType:"blob"}).then(function(e){t=e.headers["content-disposition"].split('"')[1],l()(e.data,t)})},f=function(e,t){return c.post(e,t).then(u)},h=function(e,t){return c.patch(e,t).then(u)},m=function(e){return c.delete(e).then(u)},g={login:function(e){return f("login",e)},getUsers:function(){return d("users")},getUser:function(e){return d("users/".concat(e))},getUserStartupKit:function(e,t,n){return p("users/".concat(e,"/blob"),t,n)},getClientStartupKit:function(e,t,n){return p("clients/".concat(e,"/blob"),t,n)},getOverseerStartupKit:function(e,t){return p("overseer/blob",e,t)},getServerStartupKit:function(e,t,n){return p("servers/".concat(e,"/blob"),t,n)},getClients:function(){return d("clients")},getProject:function(){return d("project")},postUser:function(e){return f("users",e)},patchUser:function(e,t){return h("users/".concat(e),t)},deleteUser:function(e){return m("users/".concat(e))},postClient:function(e){return f("clients",e)},patchClient:function(e,t){return h("clients/".concat(e),t)},deleteClient:function(e){return m("clients/".concat(e))},patchProject:function(e){return h("project",e)},getServer:function(){return d("server")},patchServer:function(e){return h("server",e)}}},12976:function(e,t,n){"use strict";n.d(t,{Z:function(){return k}});var r=n(50029),o=n(64687),i=n.n(o),s=n(41664),a=n.n(s),l=n(29224),c=n.n(l),u=n(70491),d=n.n(u),p=n(86188),f=n.n(p),h=n(85444),m=h.default.div.withConfig({displayName:"styles__StyledLayout",componentId:"sc-xczy9u-0"})(["overflow:hidden;height:100%;width:100%;margin:0;padding:0;display:flex;flex-wrap:wrap;.menu{height:auto;}.content-header{flex:0 0 80px;}"]),g=h.default.div.withConfig({displayName:"styles__StyledContent",componentId:"sc-xczy9u-1"})(["display:flex;flex-direction:column;flex:1 1 0%;overflow:auto;height:calc(100% - 3rem);.inlineeditlarger{padding:10px;}.inlineedit{padding:10px;margin:-10px;}.content-wrapper{padding:",";min-height:800px;}"],function(e){return e.theme.spacing.four}),x=n(11163),j=n(84506),v=n(67294),y=n(40203),_=n(13258),w=n.n(_),S=n(5801),b=n.n(S),C=n(14682),I=n(85893),k=function(e){var t,n=e.children,o=e.headerChildren,s=e.title,l=(0,x.useRouter)(),u=l.pathname,h=l.push,S=(0,v.useState)(),k=S[0],N=S[1],z=(0,y.Gg)();(0,v.useEffect)(function(){C.y.getProject().then(function(e){N(e.project)})},[]);var M=(t=(0,r.Z)(i().mark(function e(){return i().wrap(function(e){for(;;)switch(e.prev=e.next){case 0:(0,y.KY)("none","",-1,0),h("/");case 2:case"end":return e.stop()}},e)})),function(){return t.apply(this,arguments)});return(0,I.jsxs)(m,{children:[(0,I.jsx)(c(),{app:null==k?void 0:k.short_name,appBarActions:z.user.role>0?(0,I.jsxs)("div",{style:{display:"flex",flexDirection:"row",alignItems:"center",marginRight:10},children:[j.demo&&(0,I.jsx)("div",{children:"DEMO MODE"}),(0,I.jsx)(w(),{parentElement:(0,I.jsx)(b(),{icon:{name:"AccountCircleGenericUser",color:"white",size:22},shape:"circle",variant:"link",className:"logout-link"}),position:"top-right",children:(0,I.jsxs)(I.Fragment,{children:[(0,I.jsx)(_.ActionMenuItem,{label:"Logout",onClick:M}),!1]})})]}):(0,I.jsx)(I.Fragment,{})}),(0,I.jsxs)(f(),{className:"menu",itemMatchPattern:function(e){return e===u},itemRenderer:function(e){var t=e.title,n=e.href;return(0,I.jsx)(a(),{href:n,children:t})},location:u,children:[0==z.user.role&&(0,I.jsxs)(p.MenuContent,{children:[(0,I.jsx)(p.MenuItem,{href:"/",icon:{name:"AccountUser"},title:"Login"}),(0,I.jsx)(p.MenuItem,{href:"/registration-form",icon:{name:"ObjectsClipboardEdit"},title:"User Registration Form"})]}),4==z.user.role&&(0,I.jsxs)(p.MenuContent,{children:[(0,I.jsx)(p.MenuItem,{href:"/",icon:{name:"ViewList"},title:"Project Home"}),(0,I.jsx)(p.MenuItem,{href:"/user-dashboard",icon:{name:"ServerEdit"},title:"My Info"}),(0,I.jsx)(p.MenuItem,{href:"/project-admin-dashboard",icon:{name:"AccountGroupShieldAdd"},title:"Users Dashboard"}),(0,I.jsx)(p.MenuItem,{href:"/site-dashboard",icon:{name:"ConnectionNetworkComputers2"},title:"Client Sites"}),(0,I.jsx)(p.MenuItem,{href:"/project-configuration",icon:{name:"SettingsCog"},title:"Project Configuration"}),(0,I.jsx)(p.MenuItem,{href:"/server-config",icon:{name:"ConnectionServerNetwork1"},title:"Server Configuration"}),(0,I.jsx)(p.MenuItem,{href:"/downloads",icon:{name:"ActionsDownload"},title:"Downloads"}),(0,I.jsx)(p.MenuItem,{href:"/logout",icon:{name:"PlaybackStop"},title:"Logout"})]}),(1==z.user.role||2==z.user.role||3==z.user.role)&&(0,I.jsxs)(p.MenuContent,{children:[(0,I.jsx)(p.MenuItem,{href:"/",icon:{name:"ViewList"},title:"Project Home Page"}),(0,I.jsx)(p.MenuItem,{href:"/user-dashboard",icon:{name:"ServerEdit"},title:"My Info"}),(0,I.jsx)(p.MenuItem,{href:"/downloads",icon:{name:"ActionsDownload"},title:"Downloads"}),(0,I.jsx)(p.MenuItem,{href:"/logout",icon:{name:"PlaybackStop"},title:"Logout"})]}),(0,I.jsx)(p.MenuFooter,{})]}),(0,I.jsxs)(g,{children:[(0,I.jsx)(d(),{className:"content-header",title:s,children:o}),(0,I.jsx)("div",{className:"content-wrapper",children:n})]})]})}},89002:function(e,t,n){"use strict";n.d(t,{P:function(){return c},X:function(){return l}});var r=n(85444),o=n(36578),i=n.n(o),s=n(3159),a=n.n(s),l=(0,r.default)(a()).withConfig({displayName:"form-page__StyledFormExample",componentId:"sc-rfrcq8-0"})([".bottom{display:flex;gap:",";}.zero-left{margin-left:0;}.zero-right{margin-right:0;}"],function(e){return e.theme.spacing.four}),c=(0,r.default)(i()).withConfig({displayName:"form-page__StyledBanner",componentId:"sc-rfrcq8-1"})(["margin-bottom:1rem;"])},49570:function(e,t,n){"use strict";n.r(t);var r=n(12976),o=n(3159),i=n.n(o),s=n(67294),a=n(5801),l=n.n(a),c=n(57539),u=n.n(c),d=n(40203),p=n(89002),f=n(7731),h=n.n(f),m=n(84506),g=n(14682),x=n(11163),j=n(30038),v=n.n(j),y=n(85893);t.default=function(){var e=(0,d.Gg)(),t=(0,x.useRouter)().push,n=(0,s.useState)(!1),o=n[0],a=n[1],c=(0,s.useState)(),f=c[0],j=c[1],_=(0,s.useState)([]),w=_[0],S=_[1],b=(0,s.useState)(),C=b[0],I=b[1],k=(0,s.useState)(""),N=k[0],z=k[1],M=u();function A(){return(Math.random()+1).toString(36).substring(6).toUpperCase()}return(0,s.useEffect)(function(){if(m.demo){var n={approval_state:1,created_at:"Fri, 22 Jul 2022 16:48:27 GMT",description:"",email:"sitemanager@organization1.com",id:5,name:"John",organization:"NVIDIA",organization_id:1,password_hash:"pbkdf2:sha256:260000$93Zo4zK8kvAb6kkA$99038d7da338cdb1bc6227338f178f66cc6d2af3c471460c5b9e6c62d62c4e07",role:"org_admin",role_id:1,sites:[{name:"site-1",id:0,org:"NVIDIA",num_of_gpus:4,approval_state:100,mem_per_gpu_in_GiB:40},{name:"site-2",id:1,org:"NVIDIA",num_of_gpus:2,approval_state:0,mem_per_gpu_in_GiB:16}],updated_at:"Fri, 22 Jul 2022 16:48:27 GMT"};I(function(){return n})}else g.y.getProject().then(function(e){j(e.project)}),g.y.getUser(e.user.id).then(function(e){e&&I(e.user);var t=e.user.organization;g.y.getClients().then(function(e){S(e.client_list.filter(function(e){return""!==t&&e.organization===t}).map(function(e){var t,n;return{name:e.name,id:e.id,org:e.organization,num_of_gpus:null===(t=e.capacity)||void 0===t?void 0:t.num_of_gpus,mem_per_gpu_in_GiB:null===(n=e.capacity)||void 0===n?void 0:n.mem_per_gpu_in_GiB,approval_state:e.approval_state}}))})}).catch(function(e){console.log(e),403!==e.response.status&&t("/")})},[t,e.user.id,e.user.token,m.demo]),(0,y.jsx)(r.Z,{title:"Downloads",children:C&&f?(0,y.jsxs)(y.Fragment,{children:[(0,y.jsxs)("div",{style:{width:"46rem"},children:[100>Number(C.approval_state)&&(0,y.jsxs)(p.P,{status:"warning",rounded:!0,children:["Your registration is awaiting approval by the Project Admin. Once approved, you will be able to download your FLARE Console","org_admin"==C.role&&" and Client Site Startup Kits","."]}),!f.frozen&&(0,y.jsxs)(p.P,{status:"warning",rounded:!0,children:["The project is still being finalized by the Project Admin. Once the Project Admin has finished configuration of the project, you will be able to download your FLARE Console","org_admin"==C.role&&" and Client Site Startup Kits","."]}),(0,y.jsxs)(i(),{title:"FLARE Console",children:[(0,y.jsx)("div",{style:{fontSize:"1rem",marginTop:".25rem",marginBottom:".5rem"},children:"Download your FLARE Console to check on the status, submit jobs, and interact with the FL project:"}),f.frozen&&Number(C.approval_state)>=100?(0,y.jsx)(l(),{type:"primary",onClick:function(){a(!0);var e=A();z(e),g.y.getUserStartupKit(C.id,"flareconsole",e)},children:"Download FLARE Console"}):(0,y.jsx)(l(),{disabled:!0,children:"Download FLARE Console"})]}),""!=f.app_location&&(3==e.user.role||4==e.user.role)&&(0,y.jsxs)(i(),{title:"Downloading and configuring application docker",children:[(0,y.jsx)("div",{style:{fontSize:"1rem"},children:"To get the docker image, use the following command:"}),(0,y.jsx)(h(),{cli:!0,copyValue:"docker pull nvcr.io/nvidia/flare-train-sdk-example",children:f.app_location})]}),3==e.user.role&&(0,y.jsx)(i(),{title:"Client Sites",children:(0,y.jsx)(M,{columns:[{accessor:"name"},{accessor:"approval_state",Header:"Download",minWidth:224,Cell:function(e){var t=e.value,n=e.row;return(0,y.jsx)(y.Fragment,{children:null!=f&&f.frozen&&t&&t>=100&&Number(C.approval_state)>=100?(0,y.jsx)("div",{children:(0,y.jsx)("div",{children:(0,y.jsx)(l(),{onClick:function(){var e,t;e=parseInt(n.id),a(!0),z(t=A()),g.y.getClientStartupKit(e,"flareconsole",t)},type:"primary",children:"Download Startup Kit"})})}):(0,y.jsx)(y.Fragment,{children:(0,y.jsx)("div",{children:(0,y.jsx)("div",{children:(0,y.jsx)(l(),{disabled:!0,children:"Download Startup Kit"})})})})})}}],data:w,disableFilters:!0,disableExport:!0,getHeaderProps:function(e){return{style:{fontSize:"1rem",fontWeight:"bold"}}},getRowId:function(e,t){return""+e.id}})}),4==e.user.role&&(0,y.jsxs)(i(),{title:"Overseer and Server Startup Kits",children:[f.ha_mode?(0,y.jsx)("div",{style:{marginBottom:"0.75rem"},children:(0,y.jsx)(l(),{onClick:function(){a(!0);var e=A();z(e),g.y.getOverseerStartupKit("flareoverseer",e)},disabled:!f.frozen,type:"primary",children:"Download Overseer Startup Kit"})}):(0,y.jsx)(y.Fragment,{}),(0,y.jsx)("div",{style:{marginBottom:"0.75rem"},children:(0,y.jsx)(l(),{onClick:function(){a(!0);var e=A();z(e),g.y.getServerStartupKit(1,"flareserver1",e)},disabled:!f.frozen,type:"primary",children:"Download Server Startup Kit (primary)"})}),f.ha_mode?(0,y.jsx)("div",{children:(0,y.jsx)(l(),{onClick:function(){a(!0);var e=A();z(e),g.y.getServerStartupKit(2,"flareserver2",e)},disabled:!f.frozen,type:"primary",children:"Download Server Startup Kit (secondary)"})}):(0,y.jsx)(y.Fragment,{})]})]}),(0,y.jsx)(v(),{footer:(0,y.jsx)(y.Fragment,{children:(0,y.jsx)(l(),{type:"primary",variant:"outline",onClick:function(){a(!1)},children:"Okay"})}),size:"large",open:o,title:"Your download should be starting soon.",closeOnBackdropClick:!1,onClose:function(){a(!1)},children:(0,y.jsxs)("div",{children:["The downloaded zip file is protected by a randomly generated secure PIN. Please guard it safely. The system does not keep a copy of it.",(0,y.jsxs)("p",{children:["The secure PIN is: ",(0,y.jsx)("span",{style:{fontWeight:700,color:"#FFF",fontSize:"large"},children:N})]}),(0,y.jsx)("p",{children:"Please unzip the download with the above PIN."})]})})]}):(0,y.jsx)("span",{children:"loading..."})})}},40203:function(e,t,n){"use strict";n.d(t,{Gg:function(){return s},KY:function(){return o},a5:function(){return i}});var r={user:{email:"",token:"",id:-1,role:0},status:"unauthenticated"};function o(e,t,n,o,i){var s=arguments.length>5&&void 0!==arguments[5]&&arguments[5];return r={user:{email:e,token:t,id:n,role:o,org:i},expired:s,status:0==o?"unauthenticated":"authenticated"},localStorage.setItem("session",JSON.stringify(r)),r}function i(e){return r={user:{email:r.user.email,token:r.user.token,id:r.user.id,role:e,org:r.user.org},expired:r.expired,status:r.status},localStorage.setItem("session",JSON.stringify(r)),r}function s(){var e=localStorage.getItem("session");return null!=e&&(r=JSON.parse(e)),r}},14817:function(e,t,n){(window.__NEXT_P=window.__NEXT_P||[]).push(["/downloads",function(){return n(49570)}])},84506:function(e){"use strict";e.exports=JSON.parse('{"projectname":"New FL Project","demo":false,"url_root":"/nvflare-dashboard","arraydata":[{"name":"itemone"},{"name":"itemtwo"},{"name":"itemthree"}]}')}},function(e){e.O(0,[234,539,640,888,774,179],function(){return e(e.s=14817)}),_N_E=e.O()}]);