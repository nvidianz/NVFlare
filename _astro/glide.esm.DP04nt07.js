/*!
 * Glide.js v3.6.0
 * (c) 2013-2023 Jędrzej Chałubek (https://github.com/jedrzejchalubek/)
 * Released under the MIT License.
 */function x(e){"@babel/helpers - typeof";return typeof Symbol=="function"&&typeof Symbol.iterator=="symbol"?x=function(t){return typeof t}:x=function(t){return t&&typeof Symbol=="function"&&t.constructor===Symbol&&t!==Symbol.prototype?"symbol":typeof t},x(e)}function P(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}function et(e,t){for(var r=0;r<t.length;r++){var n=t[r];n.enumerable=n.enumerable||!1,n.configurable=!0,"value"in n&&(n.writable=!0),Object.defineProperty(e,n.key,n)}}function B(e,t,r){return t&&et(e.prototype,t),e}function nt(e,t){if(typeof t!="function"&&t!==null)throw new TypeError("Super expression must either be null or a function");e.prototype=Object.create(t&&t.prototype,{constructor:{value:e,writable:!0,configurable:!0}}),t&&M(e,t)}function O(e){return O=Object.setPrototypeOf?Object.getPrototypeOf:function(r){return r.__proto__||Object.getPrototypeOf(r)},O(e)}function M(e,t){return M=Object.setPrototypeOf||function(n,s){return n.__proto__=s,n},M(e,t)}function rt(){if(typeof Reflect>"u"||!Reflect.construct||Reflect.construct.sham)return!1;if(typeof Proxy=="function")return!0;try{return Boolean.prototype.valueOf.call(Reflect.construct(Boolean,[],function(){})),!0}catch{return!1}}function it(e){if(e===void 0)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return e}function st(e,t){if(t&&(typeof t=="object"||typeof t=="function"))return t;if(t!==void 0)throw new TypeError("Derived constructors may only return object or undefined");return it(e)}function at(e){var t=rt();return function(){var n=O(e),s;if(t){var i=O(this).constructor;s=Reflect.construct(n,arguments,i)}else s=n.apply(this,arguments);return st(this,s)}}function ot(e,t){for(;!Object.prototype.hasOwnProperty.call(e,t)&&(e=O(e),e!==null););return e}function z(){return typeof Reflect<"u"&&Reflect.get?z=Reflect.get:z=function(t,r,n){var s=ot(t,r);if(s){var i=Object.getOwnPropertyDescriptor(s,r);return i.get?i.get.call(arguments.length<3?t:n):i.value}},z.apply(this,arguments)}var ut={type:"slider",startAt:0,perView:1,focusAt:0,gap:10,autoplay:!1,hoverpause:!0,keyboard:!0,bound:!1,swipeThreshold:80,dragThreshold:120,perSwipe:"",touchRatio:.5,touchAngle:45,animationDuration:400,rewind:!0,rewindDuration:800,animationTimingFunc:"cubic-bezier(.165, .840, .440, 1)",waitForTransition:!0,throttle:10,direction:"ltr",peek:0,cloningRatio:1,breakpoints:{},lazy:!1,lazyScrollThreshold:1.2,lazyInitialSlidesLoaded:2,classes:{swipeable:"glide--swipeable",dragging:"glide--dragging",direction:{ltr:"glide--ltr",rtl:"glide--rtl"},type:{slider:"glide--slider",carousel:"glide--carousel"},slide:{clone:"glide__slide--clone",active:"glide__slide--active"},arrow:{disabled:"glide__arrow--disabled"},nav:{active:"glide__bullet--active"}}};function y(e){console.error("[Glide warn]: ".concat(e))}function m(e){return parseInt(e)}function ct(e){return parseFloat(e)}function I(e){return typeof e=="string"}function k(e){var t=x(e);return t==="function"||t==="object"&&!!e}function L(e){return typeof e=="function"}function Q(e){return typeof e>"u"}function D(e){return e.constructor===Array}function ft(e,t,r){var n={};for(var s in t)L(t[s])?n[s]=t[s](e,n,r):y("Extension must be a function");for(var i in n)L(n[i].mount)&&n[i].mount();return n}function h(e,t,r){Object.defineProperty(e,t,r)}function lt(e){return Object.keys(e).sort().reduce(function(t,r){return t[r]=e[r],t[r],t},{})}function V(e,t){var r=Object.assign({},e,t);return t.hasOwnProperty("classes")&&(r.classes=Object.assign({},e.classes,t.classes),t.classes.hasOwnProperty("direction")&&(r.classes.direction=Object.assign({},e.classes.direction,t.classes.direction)),t.classes.hasOwnProperty("type")&&(r.classes.type=Object.assign({},e.classes.type,t.classes.type)),t.classes.hasOwnProperty("slide")&&(r.classes.slide=Object.assign({},e.classes.slide,t.classes.slide)),t.classes.hasOwnProperty("arrow")&&(r.classes.arrow=Object.assign({},e.classes.arrow,t.classes.arrow)),t.classes.hasOwnProperty("nav")&&(r.classes.nav=Object.assign({},e.classes.nav,t.classes.nav))),t.hasOwnProperty("breakpoints")&&(r.breakpoints=Object.assign({},e.breakpoints,t.breakpoints)),r}var dt=function(){function e(){var t=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{};P(this,e),this.events=t,this.hop=t.hasOwnProperty}return B(e,[{key:"on",value:function(r,n){if(D(r)){for(var s=0;s<r.length;s++)this.on(r[s],n);return}this.hop.call(this.events,r)||(this.events[r]=[]);var i=this.events[r].push(n)-1;return{remove:function(){delete this.events[r][i]}}}},{key:"emit",value:function(r,n){if(D(r)){for(var s=0;s<r.length;s++)this.emit(r[s],n);return}this.hop.call(this.events,r)&&this.events[r].forEach(function(i){i(n||{})})}}]),e}(),ht=function(){function e(t){var r=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{};P(this,e),this._c={},this._t=[],this._e=new dt,this.disabled=!1,this.selector=t,this.settings=V(ut,r),this.index=this.settings.startAt}return B(e,[{key:"mount",value:function(){var r=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{};return this._e.emit("mount.before"),k(r)?this._c=ft(this,r,this._e):y("You need to provide a object on `mount()`"),this._e.emit("mount.after"),this}},{key:"mutate",value:function(){var r=arguments.length>0&&arguments[0]!==void 0?arguments[0]:[];return D(r)?this._t=r:y("You need to provide a array on `mutate()`"),this}},{key:"update",value:function(){var r=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{};return this.settings=V(this.settings,r),r.hasOwnProperty("startAt")&&(this.index=r.startAt),this._e.emit("update"),this}},{key:"go",value:function(r){return this._c.Run.make(r),this}},{key:"move",value:function(r){return this._c.Transition.disable(),this._c.Move.make(r),this}},{key:"destroy",value:function(){return this._e.emit("destroy"),this}},{key:"play",value:function(){var r=arguments.length>0&&arguments[0]!==void 0?arguments[0]:!1;return r&&(this.settings.autoplay=r),this._e.emit("play"),this}},{key:"pause",value:function(){return this._e.emit("pause"),this}},{key:"disable",value:function(){return this.disabled=!0,this}},{key:"enable",value:function(){return this.disabled=!1,this}},{key:"on",value:function(r,n){return this._e.on(r,n),this}},{key:"isType",value:function(r){return this.settings.type===r}},{key:"settings",get:function(){return this._o},set:function(r){k(r)?this._o=r:y("Options must be an `object` instance.")}},{key:"index",get:function(){return this._i},set:function(r){this._i=m(r)}},{key:"type",get:function(){return this.settings.type}},{key:"disabled",get:function(){return this._d},set:function(r){this._d=!!r}}]),e}();function vt(e,t,r){var n={mount:function(){this._o=!1},make:function(f){var c=this;e.disabled||(!e.settings.waitForTransition||e.disable(),this.move=f,r.emit("run.before",this.move),this.calculate(),r.emit("run",this.move),t.Transition.after(function(){c.isStart()&&r.emit("run.start",c.move),c.isEnd()&&r.emit("run.end",c.move),c.isOffset()&&(c._o=!1,r.emit("run.offset",c.move)),r.emit("run.after",c.move),e.enable()}))},calculate:function(){var f=this.move,c=this.length,d=f.steps,l=f.direction,v=1;if(l==="="){if(e.settings.bound&&m(d)>c){e.index=c;return}e.index=d;return}if(l===">"&&d===">"){e.index=c;return}if(l==="<"&&d==="<"){e.index=0;return}if(l==="|"&&(v=e.settings.perView||1),l===">"||l==="|"&&d===">"){var p=s(v);p>c&&(this._o=!0),e.index=i(p,v);return}if(l==="<"||l==="|"&&d==="<"){var g=a(v);g<0&&(this._o=!0),e.index=o(g,v);return}y("Invalid direction pattern [".concat(l).concat(d,"] has been used"))},isStart:function(){return e.index<=0},isEnd:function(){return e.index>=this.length},isOffset:function(){var f=arguments.length>0&&arguments[0]!==void 0?arguments[0]:void 0;return f?this._o?f==="|>"?this.move.direction==="|"&&this.move.steps===">":f==="|<"?this.move.direction==="|"&&this.move.steps==="<":this.move.direction===f:!1:this._o},isBound:function(){return e.isType("slider")&&e.settings.focusAt!=="center"&&e.settings.bound}};function s(u){var f=e.index;return e.isType("carousel")?f+u:f+(u-f%u)}function i(u,f){var c=n.length;return u<=c?u:e.isType("carousel")?u-(c+1):e.settings.rewind?n.isBound()&&!n.isEnd()?c:0:n.isBound()?c:Math.floor(c/f)*f}function a(u){var f=e.index;if(e.isType("carousel"))return f-u;var c=Math.ceil(f/u);return(c-1)*u}function o(u,f){var c=n.length;return u>=0?u:e.isType("carousel")?u+(c+1):e.settings.rewind?n.isBound()&&n.isStart()?c:Math.floor(c/f)*f:0}return h(n,"move",{get:function(){return this._m},set:function(f){var c=f.substr(1);this._m={direction:f.substr(0,1),steps:c?m(c)?m(c):c:0}}}),h(n,"length",{get:function(){var f=e.settings,c=t.Html.slides.length;return this.isBound()?c-1-(m(f.perView)-1)+m(f.focusAt):c-1}}),h(n,"offset",{get:function(){return this._o}}),n}function F(){return new Date().getTime()}function R(e,t,r){var n,s,i,a,o=0;r||(r={});var u=function(){o=r.leading===!1?0:F(),n=null,a=e.apply(s,i),n||(s=i=null)},f=function(){var d=F();!o&&r.leading===!1&&(o=d);var l=t-(d-o);return s=this,i=arguments,l<=0||l>t?(n&&(clearTimeout(n),n=null),o=d,a=e.apply(s,i),n||(s=i=null)):!n&&r.trailing!==!1&&(n=setTimeout(u,l)),a};return f.cancel=function(){clearTimeout(n),o=0,n=s=i=null},f}var A={ltr:["marginLeft","marginRight"],rtl:["marginRight","marginLeft"]};function gt(e,t,r){var n={apply:function(i){for(var a=0,o=i.length;a<o;a++){var u=i[a].style,f=t.Direction.value;a!==0?u[A[f][0]]="".concat(this.value/2,"px"):u[A[f][0]]="",a!==i.length-1?u[A[f][1]]="".concat(this.value/2,"px"):u[A[f][1]]=""}},remove:function(i){for(var a=0,o=i.length;a<o;a++){var u=i[a].style;u.marginLeft="",u.marginRight=""}}};return h(n,"value",{get:function(){return m(e.settings.gap)}}),h(n,"grow",{get:function(){return n.value*t.Sizes.length}}),h(n,"reductor",{get:function(){var i=e.settings.perView;return n.value*(i-1)/i}}),r.on(["build.after","update"],R(function(){n.apply(t.Html.wrapper.children)},30)),r.on("destroy",function(){n.remove(t.Html.wrapper.children)}),n}function Z(e){if(e&&e.parentNode){for(var t=e.parentNode.firstChild,r=[];t;t=t.nextSibling)t.nodeType===1&&t!==e&&r.push(t);return r}return[]}function q(e){return!!(e&&e instanceof window.HTMLElement)}function E(e){return Array.prototype.slice.call(e)}var K='[data-glide-el="track"]';function mt(e,t,r){var n={mount:function(){this.root=e.selector,this.track=this.root.querySelector(K),this.collectSlides()},collectSlides:function(){this.slides=E(this.wrapper.children).filter(function(i){return!i.classList.contains(e.settings.classes.slide.clone)})}};return h(n,"root",{get:function(){return n._r},set:function(i){I(i)&&(i=document.querySelector(i)),q(i)?n._r=i:y("Root element must be a existing Html node")}}),h(n,"track",{get:function(){return n._t},set:function(i){q(i)?n._t=i:y("Could not find track element. Please use ".concat(K," attribute."))}}),h(n,"wrapper",{get:function(){return n.track.children[0]}}),r.on("update",function(){n.collectSlides()}),n}function pt(e,t,r){var n={mount:function(){this.value=e.settings.peek}};return h(n,"value",{get:function(){return n._v},set:function(i){k(i)?(i.before=m(i.before),i.after=m(i.after)):i=m(i),n._v=i}}),h(n,"reductor",{get:function(){var i=n.value,a=e.settings.perView;return k(i)?i.before/a+i.after/a:i*2/a}}),r.on(["resize","update"],function(){n.mount()}),n}function yt(e,t,r){var n={mount:function(){this._o=0},make:function(){var i=this,a=arguments.length>0&&arguments[0]!==void 0?arguments[0]:0;this.offset=a,r.emit("move",{movement:this.value}),t.Transition.after(function(){r.emit("move.after",{movement:i.value})})}};return h(n,"offset",{get:function(){return n._o},set:function(i){n._o=Q(i)?0:m(i)}}),h(n,"translate",{get:function(){return t.Sizes.slideWidth*e.index}}),h(n,"value",{get:function(){var i=this.offset,a=this.translate;return t.Direction.is("rtl")?a+i:a-i}}),r.on(["build.before","run"],function(){n.make()}),n}function bt(e,t,r){var n={setupSlides:function(){for(var i="".concat(this.slideWidth,"px"),a=t.Html.slides,o=0;o<a.length;o++)a[o].style.width=i},setupWrapper:function(){t.Html.wrapper.style.width="".concat(this.wrapperSize,"px")},remove:function(){for(var i=t.Html.slides,a=0;a<i.length;a++)i[a].style.width="";t.Html.wrapper.style.width=""}};return h(n,"length",{get:function(){return t.Html.slides.length}}),h(n,"width",{get:function(){return t.Html.track.offsetWidth}}),h(n,"wrapperSize",{get:function(){return n.slideWidth*n.length+t.Gaps.grow+t.Clones.grow}}),h(n,"slideWidth",{get:function(){return n.width/e.settings.perView-t.Peek.reductor-t.Gaps.reductor}}),r.on(["build.before","resize","update"],function(){n.setupSlides(),n.setupWrapper()}),r.on("destroy",function(){n.remove()}),n}function wt(e,t,r){var n={mount:function(){r.emit("build.before"),this.typeClass(),this.activeClass(),r.emit("build.after")},typeClass:function(){t.Html.root.classList.add(e.settings.classes.type[e.settings.type])},activeClass:function(){var i=e.settings.classes,a=t.Html.slides[e.index];a&&(a.classList.add(i.slide.active),Z(a).forEach(function(o){o.classList.remove(i.slide.active)}))},removeClasses:function(){var i=e.settings.classes,a=i.type,o=i.slide;t.Html.root.classList.remove(a[e.settings.type]),t.Html.slides.forEach(function(u){u.classList.remove(o.active)})}};return r.on(["destroy","update"],function(){n.removeClasses()}),r.on(["resize","update"],function(){n.mount()}),r.on("move.after",function(){n.activeClass()}),n}function _t(e,t,r){var n={mount:function(){this.items=[],e.isType("carousel")&&(this.items=this.collect())},collect:function(){var i=arguments.length>0&&arguments[0]!==void 0?arguments[0]:[],a=t.Html.slides,o=e.settings,u=o.perView,f=o.classes,c=o.cloningRatio;if(a.length!==0)for(var d=+!!e.settings.peek,l=u+d+Math.round(u/2),v=a.slice(0,l).reverse(),p=a.slice(l*-1),g=0;g<Math.max(c,Math.floor(u/a.length));g++){for(var b=0;b<v.length;b++){var w=v[b].cloneNode(!0);w.classList.add(f.slide.clone),i.push(w)}for(var _=0;_<p.length;_++){var T=p[_].cloneNode(!0);T.classList.add(f.slide.clone),i.unshift(T)}}return i},append:function(){for(var i=this.items,a=t.Html,o=a.wrapper,u=a.slides,f=Math.floor(i.length/2),c=i.slice(0,f).reverse(),d=i.slice(f*-1).reverse(),l="".concat(t.Sizes.slideWidth,"px"),v=0;v<d.length;v++)o.appendChild(d[v]);for(var p=0;p<c.length;p++)o.insertBefore(c[p],u[0]);for(var g=0;g<i.length;g++)i[g].style.width=l},remove:function(){for(var i=this.items,a=0;a<i.length;a++)t.Html.wrapper.removeChild(i[a])}};return h(n,"grow",{get:function(){return(t.Sizes.slideWidth+t.Gaps.value)*n.items.length}}),r.on("update",function(){n.remove(),n.mount(),n.append()}),r.on("build.before",function(){e.isType("carousel")&&n.append()}),r.on("destroy",function(){n.remove()}),n}var S=function(){function e(){var t=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{};P(this,e),this.listeners=t}return B(e,[{key:"on",value:function(r,n,s){var i=arguments.length>3&&arguments[3]!==void 0?arguments[3]:!1;I(r)&&(r=[r]);for(var a=0;a<r.length;a++)this.listeners[r[a]]=s,n.addEventListener(r[a],this.listeners[r[a]],i)}},{key:"off",value:function(r,n){var s=arguments.length>2&&arguments[2]!==void 0?arguments[2]:!1;I(r)&&(r=[r]);for(var i=0;i<r.length;i++)n.removeEventListener(r[i],this.listeners[r[i]],s)}},{key:"destroy",value:function(){delete this.listeners}}]),e}();function St(e,t,r){var n=new S,s={mount:function(){this.bind()},bind:function(){n.on("resize",window,R(function(){r.emit("resize")},e.settings.throttle))},unbind:function(){n.off("resize",window)}};return r.on("destroy",function(){s.unbind(),n.destroy()}),s}var Tt=["ltr","rtl"],Ot={">":"<","<":">","=":"="};function kt(e,t,r){var n={mount:function(){this.value=e.settings.direction},resolve:function(i){var a=i.slice(0,1);return this.is("rtl")?i.split(a).join(Ot[a]):i},is:function(i){return this.value===i},addClass:function(){t.Html.root.classList.add(e.settings.classes.direction[this.value])},removeClass:function(){t.Html.root.classList.remove(e.settings.classes.direction[this.value])}};return h(n,"value",{get:function(){return n._v},set:function(i){Tt.indexOf(i)>-1?n._v=i:y("Direction value must be `ltr` or `rtl`")}}),r.on(["destroy","update"],function(){n.removeClass()}),r.on("update",function(){n.mount()}),r.on(["build.before","update"],function(){n.addClass()}),n}function Rt(e,t){return{modify:function(n){return t.Direction.is("rtl")?-n:n}}}function At(e,t){return{modify:function(n){var s=Math.floor(n/t.Sizes.slideWidth);return n+t.Gaps.value*s}}}function Ht(e,t){return{modify:function(n){return n+t.Clones.grow/2}}}function xt(e,t){return{modify:function(n){if(e.settings.focusAt>=0){var s=t.Peek.value;return k(s)?n-s.before:n-s}return n}}}function zt(e,t){return{modify:function(n){var s=t.Gaps.value,i=t.Sizes.width,a=e.settings.focusAt,o=t.Sizes.slideWidth;return a==="center"?n-(i/2-o/2):n-o*a-s*a}}}function Lt(e,t,r){var n=[At,Ht,xt,zt].concat(e._t,[Rt]);return{mutate:function(i){for(var a=0;a<n.length;a++){var o=n[a];L(o)&&L(o().modify)?i=o(e,t,r).modify(i):y("Transformer should be a function that returns an object with `modify()` method")}return i}}}function Pt(e,t,r){var n={set:function(i){var a=Lt(e,t).mutate(i),o="translate3d(".concat(-1*a,"px, 0px, 0px)");t.Html.wrapper.style.mozTransform=o,t.Html.wrapper.style.webkitTransform=o,t.Html.wrapper.style.transform=o},remove:function(){t.Html.wrapper.style.transform=""},getStartIndex:function(){var i=t.Sizes.length,a=e.index,o=e.settings.perView;return t.Run.isOffset(">")||t.Run.isOffset("|>")?i+(a-o):(a+o)%i},getTravelDistance:function(){var i=t.Sizes.slideWidth*e.settings.perView;return t.Run.isOffset(">")||t.Run.isOffset("|>")?i*-1:i}};return r.on("move",function(s){if(!e.isType("carousel")||!t.Run.isOffset())return n.set(s.movement);t.Transition.after(function(){r.emit("translate.jump"),n.set(t.Sizes.slideWidth*e.index)});var i=t.Sizes.slideWidth*t.Translate.getStartIndex();return n.set(i-t.Translate.getTravelDistance())}),r.on("destroy",function(){n.remove()}),n}function Bt(e,t,r){var n=!1,s={compose:function(a){var o=e.settings;return n?"".concat(a," 0ms ").concat(o.animationTimingFunc):"".concat(a," ").concat(this.duration,"ms ").concat(o.animationTimingFunc)},set:function(){var a=arguments.length>0&&arguments[0]!==void 0?arguments[0]:"transform";t.Html.wrapper.style.transition=this.compose(a)},remove:function(){t.Html.wrapper.style.transition=""},after:function(a){setTimeout(function(){a()},this.duration)},enable:function(){n=!1,this.set()},disable:function(){n=!0,this.set()}};return h(s,"duration",{get:function(){var a=e.settings;return e.isType("slider")&&t.Run.offset?a.rewindDuration:a.animationDuration}}),r.on("move",function(){s.set()}),r.on(["build.before","resize","translate.jump"],function(){s.disable()}),r.on("run",function(){s.enable()}),r.on("destroy",function(){s.remove()}),s}var C=!1;try{var X=Object.defineProperty({},"passive",{get:function(){C=!0}});window.addEventListener("testPassive",null,X),window.removeEventListener("testPassive",null,X)}catch{}var j=C,H=["touchstart","mousedown"],Y=["touchmove","mousemove"],$=["touchend","touchcancel","mouseup","mouseleave"],U=["mousedown","mousemove","mouseup","mouseleave"];function Mt(e,t,r){var n=new S,s=0,i=0,a=0,o=!1,u=j?{passive:!0}:!1,f={mount:function(){this.bindSwipeStart()},start:function(d){if(!o&&!e.disabled){this.disable();var l=this.touches(d);s=null,i=m(l.pageX),a=m(l.pageY),this.bindSwipeMove(),this.bindSwipeEnd(),r.emit("swipe.start")}},move:function(d){if(!e.disabled){var l=e.settings,v=l.touchAngle,p=l.touchRatio,g=l.classes,b=this.touches(d),w=m(b.pageX)-i,_=m(b.pageY)-a,T=Math.abs(w<<2),W=Math.abs(_<<2),G=Math.sqrt(T+W),tt=Math.sqrt(W);if(s=Math.asin(tt/G),s*180/Math.PI<v)d.stopPropagation(),t.Move.make(w*ct(p)),t.Html.root.classList.add(g.dragging),r.emit("swipe.move");else return!1}},end:function(d){if(!e.disabled){var l=e.settings,v=l.perSwipe,p=l.touchAngle,g=l.classes,b=this.touches(d),w=this.threshold(d),_=b.pageX-i,T=s*180/Math.PI;this.enable(),_>w&&T<p?t.Run.make(t.Direction.resolve("".concat(v,"<"))):_<-w&&T<p?t.Run.make(t.Direction.resolve("".concat(v,">"))):t.Move.make(),t.Html.root.classList.remove(g.dragging),this.unbindSwipeMove(),this.unbindSwipeEnd(),r.emit("swipe.end")}},bindSwipeStart:function(){var d=this,l=e.settings,v=l.swipeThreshold,p=l.dragThreshold;v&&n.on(H[0],t.Html.wrapper,function(g){d.start(g)},u),p&&n.on(H[1],t.Html.wrapper,function(g){d.start(g)},u)},unbindSwipeStart:function(){n.off(H[0],t.Html.wrapper,u),n.off(H[1],t.Html.wrapper,u)},bindSwipeMove:function(){var d=this;n.on(Y,t.Html.wrapper,R(function(l){d.move(l)},e.settings.throttle),u)},unbindSwipeMove:function(){n.off(Y,t.Html.wrapper,u)},bindSwipeEnd:function(){var d=this;n.on($,t.Html.wrapper,function(l){d.end(l)})},unbindSwipeEnd:function(){n.off($,t.Html.wrapper)},touches:function(d){return U.indexOf(d.type)>-1?d:d.touches[0]||d.changedTouches[0]},threshold:function(d){var l=e.settings;return U.indexOf(d.type)>-1?l.dragThreshold:l.swipeThreshold},enable:function(){return o=!1,t.Transition.enable(),this},disable:function(){return o=!0,t.Transition.disable(),this}};return r.on("build.after",function(){t.Html.root.classList.add(e.settings.classes.swipeable)}),r.on("destroy",function(){f.unbindSwipeStart(),f.unbindSwipeMove(),f.unbindSwipeEnd(),n.destroy()}),f}function It(e,t,r){var n=new S,s={mount:function(){this.bind()},bind:function(){n.on("dragstart",t.Html.wrapper,this.dragstart)},unbind:function(){n.off("dragstart",t.Html.wrapper)},dragstart:function(a){a.preventDefault()}};return r.on("destroy",function(){s.unbind(),n.destroy()}),s}function Dt(e,t,r){var n=e.settings,s=!1,i={mount:function(){n.lazy&&(this._wrapper=t.Html.root,this._slideElements=this._wrapper.querySelectorAll(".glide__slide"))},withinView:function(){var o=this._wrapper.getBoundingClientRect();o.bottom>0&&o.right>0&&o.top<=(window.innerHeight*n.lazyScrollThreshold||document.documentElement.clientHeight)*n.lazyScrollThreshold&&o.left<=(window.innerWidth*n.lazyScrollThreshold||document.documentElement.clientWidth*n.lazyScrollThreshold)&&this.lazyLoad()},lazyLoad:function(){var o,u=n.lazyInitialSlidesLoaded-1;s=!0,e.index+u<this._slideElements.length?o=e.index+u:o=e.index;for(var f=0;f<=o;f++){var c=this._slideElements[f].getElementsByTagName("img")[0];c&&c.classList.contains("glide__lazy")&&(this._slideElements[f].classList.contains("glide__lazy__loaded")||this.loadImage(c))}},loadImage:function(o){o.dataset.src&&(o.src=o.dataset.src,o.classList.add("glide__lazy__loaded"),o.classList.remove("glide__lazy"),o.removeAttribute("data-src"))}};return r.on(["mount.after"],function(){n.lazy&&i.withinView()}),r.on(["move.after"],R(function(){n.lazy&&s?i.lazyLoad():n.lazy&&i.withinView()},100)),document.addEventListener("scroll",R(function(){n.lazy&&!s&&i.withinView()},100)),i}function Vt(e,t,r){var n=new S,s=!1,i=!1,a={mount:function(){this._a=t.Html.wrapper.querySelectorAll("a"),this.bind()},bind:function(){n.on("click",t.Html.wrapper,this.click)},unbind:function(){n.off("click",t.Html.wrapper)},click:function(u){i&&(u.stopPropagation(),u.preventDefault())},detach:function(){if(i=!0,!s){for(var u=0;u<this.items.length;u++)this.items[u].draggable=!1;s=!0}return this},attach:function(){if(i=!1,s){for(var u=0;u<this.items.length;u++)this.items[u].draggable=!0;s=!1}return this}};return h(a,"items",{get:function(){return a._a}}),r.on("swipe.move",function(){a.detach()}),r.on("swipe.end",function(){t.Transition.after(function(){a.attach()})}),r.on("destroy",function(){a.attach(),a.unbind(),n.destroy()}),a}var Et='[data-glide-el="controls[nav]"]',N='[data-glide-el^="controls"]',jt="".concat(N,' [data-glide-dir*="<"]'),Nt="".concat(N,' [data-glide-dir*=">"]');function Wt(e,t,r){var n=new S,s=j?{passive:!0}:!1,i={mount:function(){this._n=t.Html.root.querySelectorAll(Et),this._c=t.Html.root.querySelectorAll(N),this._arrowControls={previous:t.Html.root.querySelectorAll(jt),next:t.Html.root.querySelectorAll(Nt)},this.addBindings()},setActive:function(){for(var o=0;o<this._n.length;o++)this.addClass(this._n[o].children)},removeActive:function(){for(var o=0;o<this._n.length;o++)this.removeClass(this._n[o].children)},addClass:function(o){var u=e.settings,f=o[e.index];f&&f&&(f.classList.add(u.classes.nav.active),Z(f).forEach(function(c){c.classList.remove(u.classes.nav.active)}))},removeClass:function(o){var u=o[e.index];u&&u.classList.remove(e.settings.classes.nav.active)},setArrowState:function(){if(!e.settings.rewind){var o=i._arrowControls.next,u=i._arrowControls.previous;this.resetArrowState(o,u),e.index===0&&this.disableArrow(u),e.index===t.Run.length&&this.disableArrow(o)}},resetArrowState:function(){for(var o=e.settings,u=arguments.length,f=new Array(u),c=0;c<u;c++)f[c]=arguments[c];f.forEach(function(d){E(d).forEach(function(l){l.classList.remove(o.classes.arrow.disabled)})})},disableArrow:function(){for(var o=e.settings,u=arguments.length,f=new Array(u),c=0;c<u;c++)f[c]=arguments[c];f.forEach(function(d){E(d).forEach(function(l){l.classList.add(o.classes.arrow.disabled)})})},addBindings:function(){for(var o=0;o<this._c.length;o++)this.bind(this._c[o].children)},removeBindings:function(){for(var o=0;o<this._c.length;o++)this.unbind(this._c[o].children)},bind:function(o){for(var u=0;u<o.length;u++)n.on("click",o[u],this.click),n.on("touchstart",o[u],this.click,s)},unbind:function(o){for(var u=0;u<o.length;u++)n.off(["click","touchstart"],o[u])},click:function(o){!j&&o.type==="touchstart"&&o.preventDefault();var u=o.currentTarget.getAttribute("data-glide-dir");t.Run.make(t.Direction.resolve(u))}};return h(i,"items",{get:function(){return i._c}}),r.on(["mount.after","move.after"],function(){i.setActive()}),r.on(["mount.after","run"],function(){i.setArrowState()}),r.on("destroy",function(){i.removeBindings(),i.removeActive(),n.destroy()}),i}function Ft(e,t,r){var n=new S,s={mount:function(){e.settings.keyboard&&this.bind()},bind:function(){n.on("keyup",document,this.press)},unbind:function(){n.off("keyup",document)},press:function(a){var o=e.settings.perSwipe;a.code==="ArrowRight"&&t.Run.make(t.Direction.resolve("".concat(o,">"))),a.code==="ArrowLeft"&&t.Run.make(t.Direction.resolve("".concat(o,"<")))}};return r.on(["destroy","update"],function(){s.unbind()}),r.on("update",function(){s.mount()}),r.on("destroy",function(){n.destroy()}),s}function qt(e,t,r){var n=new S,s={mount:function(){this.enable(),this.start(),e.settings.hoverpause&&this.bind()},enable:function(){this._e=!0},disable:function(){this._e=!1},start:function(){var a=this;this._e&&(this.enable(),e.settings.autoplay&&Q(this._i)&&(this._i=setInterval(function(){a.stop(),t.Run.make(">"),a.start(),r.emit("autoplay")},this.time)))},stop:function(){this._i=clearInterval(this._i)},bind:function(){var a=this;n.on("mouseover",t.Html.root,function(){a._e&&a.stop()}),n.on("mouseout",t.Html.root,function(){a._e&&a.start()})},unbind:function(){n.off(["mouseover","mouseout"],t.Html.root)}};return h(s,"time",{get:function(){var a=t.Html.slides[e.index].getAttribute("data-glide-autoplay");return m(a||e.settings.autoplay)}}),r.on(["destroy","update"],function(){s.unbind()}),r.on(["run.before","swipe.start","update"],function(){s.stop()}),r.on(["pause","destroy"],function(){s.disable(),s.stop()}),r.on(["run.after","swipe.end"],function(){s.start()}),r.on(["play"],function(){s.enable(),s.start()}),r.on("update",function(){s.mount()}),r.on("destroy",function(){n.destroy()}),s}function J(e){return k(e)?lt(e):(y("Breakpoints option must be an object"),{})}function Kt(e,t,r){var n=new S,s=e.settings,i=J(s.breakpoints),a=Object.assign({},s),o={match:function(f){if(typeof window.matchMedia<"u"){for(var c in f)if(f.hasOwnProperty(c)&&window.matchMedia("(max-width: ".concat(c,"px)")).matches)return f[c]}return a}};return Object.assign(s,o.match(i)),n.on("resize",window,R(function(){e.settings=V(s,o.match(i))},e.settings.throttle)),r.on("update",function(){i=J(i),a=Object.assign({},s)}),r.on("destroy",function(){n.off("resize",window)}),o}var Xt={Html:mt,Translate:Pt,Transition:Bt,Direction:kt,Peek:pt,Sizes:bt,Gaps:gt,Move:yt,Clones:_t,Resize:St,Build:wt,Run:vt,Swipe:Mt,Images:It,Anchors:Vt,Controls:Wt,Keyboard:Ft,Autoplay:qt,Breakpoints:Kt,Lazy:Dt},Yt=function(e){nt(r,e);var t=at(r);function r(){return P(this,r),t.apply(this,arguments)}return B(r,[{key:"mount",value:function(){var s=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{};return z(O(r.prototype),"mount",this).call(this,Object.assign({},Xt,s))}}]),r}(ht);export{Yt as G};
