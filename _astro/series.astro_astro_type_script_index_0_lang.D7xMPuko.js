import{G as d}from"./glide.esm.DP04nt07.js";const s={type:"slider",animationDuration:300,rewindDuration:300,perView:1,gap:32,dragThreshold:!1,touchRatio:0,swipeThreshold:!1},e=document.getElementById("series-glide");var n=new d(e,s).mount(),o=e.querySelectorAll(".glide__slide");n.on("move.after",function(l){o.forEach(i=>{var t=document.getElementById(i.id+"-content");i.classList.contains("glide__slide--active")?t?.classList.remove("hidden"):t?.classList.add("hidden")})});e.addEventListener("click",a);function a(){n.update()}
