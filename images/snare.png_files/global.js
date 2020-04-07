(function(a,b){if("function"==typeof define&&define.amd)define([],b);else if("undefined"!=typeof exports)b();else{b(),a.FileSaver={exports:{}}.exports}})(this,function(){"use strict";function b(a,b){return"undefined"==typeof b?b={autoBom:!1}:"object"!=typeof b&&(console.warn("Depricated: Expected third argument to be a object"),b={autoBom:!b}),b.autoBom&&/^\s*(?:text\/\S*|application\/xml|\S*\/\S*\+xml)\s*;.*charset\s*=\s*utf-8/i.test(a.type)?new Blob(["\uFEFF",a],{type:a.type}):a}function c(b,c,d){var e=new XMLHttpRequest;e.open("GET",b),e.responseType="blob",e.onload=function(){a(e.response,c,d)},e.onerror=function(){console.error("could not download file")},e.send()}function d(a){var b=new XMLHttpRequest;return b.open("HEAD",a,!1),b.send(),200<=b.status&&299>=b.status}function e(a){try{a.dispatchEvent(new MouseEvent("click"))}catch(c){var b=document.createEvent("MouseEvents");b.initMouseEvent("click",!0,!0,window,0,0,0,80,20,!1,!1,!1,!1,0,null),a.dispatchEvent(b)}}var f="object"==typeof window&&window.window===window?window:"object"==typeof self&&self.self===self?self:"object"==typeof global&&global.global===global?global:void 0,a=f.saveAs||("object"!=typeof window||window!==f?function(){}:"download"in HTMLAnchorElement.prototype?function(b,g,h){var i=f.URL||f.webkitURL,j=document.createElement("a");g=g||b.name||"download",j.download=g,j.rel="noopener","string"==typeof b?(j.href=b,j.origin===location.origin?e(j):d(j.href)?c(b,g,h):e(j,j.target="_blank")):(j.href=i.createObjectURL(b),setTimeout(function(){i.revokeObjectURL(j.href)},4E4),setTimeout(function(){e(j)},0))}:"msSaveOrOpenBlob"in navigator?function(f,g,h){if(g=g||f.name||"download","string"!=typeof f)navigator.msSaveOrOpenBlob(b(f,h),g);else if(d(f))c(f,g,h);else{var i=document.createElement("a");i.href=f,i.target="_blank",setTimeout(function(){e(i)})}}:function(a,b,d,e){if(e=e||open("","_blank"),e&&(e.document.title=e.document.body.innerText="downloading..."),"string"==typeof a)return c(a,b,d);var g="application/octet-stream"===a.type,h=/constructor/i.test(f.HTMLElement)||f.safari,i=/CriOS\/[\d]+/.test(navigator.userAgent);if((i||g&&h)&&"object"==typeof FileReader){var j=new FileReader;j.onloadend=function(){var a=j.result;a=i?a:a.replace(/^data:[^;]*;/,"data:attachment/file;"),e?e.location.href=a:location=a,e=null},j.readAsDataURL(a)}else{var k=f.URL||f.webkitURL,l=k.createObjectURL(a);e?e.location=l:location.href=l,e=null,setTimeout(function(){k.revokeObjectURL(l)},4E4)}});f.saveAs=a.saveAs=a,"undefined"!=typeof module&&(module.exports=a)});
!function(t,e){function n(t){return t&&e.XDomainRequest&&!/MSIE 1/.test(navigator.userAgent)?new XDomainRequest:e.XMLHttpRequest?new XMLHttpRequest:void 0}function o(t,e,n){t[e]=t[e]||n}var r=["responseType","withCredentials","timeout","onprogress"];t.ajax=function(t,a){function s(t,e){return function(){c||(a(void 0===f.status?t:f.status,0===f.status?"Error":f.response||f.responseText||e,f),c=!0)}}var u=t.headers||{},i=t.body,d=t.method||(i?"POST":"GET"),c=!1,f=n(t.cors);f.open(d,t.url,!0);var l=f.onload=s(200);f.onreadystatechange=function(){4===f.readyState&&l()},f.onerror=s(null,"Error"),f.ontimeout=s(null,"Timeout"),f.onabort=s(null,"Abort"),i&&(o(u,"X-Requested-With","XMLHttpRequest"),e.FormData&&i instanceof e.FormData||o(u,"Content-Type","application/x-www-form-urlencoded"));for(var p,m=0,v=r.length;v>m;m++)p=r[m],void 0!==t[p]&&(f[p]=t[p]);for(var p in u)f.setRequestHeader(p,u[p]);return f.send(i),f},e.nanoajax=t}({},function(){return this}());
Element.prototype.remove=function(){this.parentElement.removeChild(this)};NodeList.prototype.remove=HTMLCollection.prototype.remove=function(){for(var a=this.length-1;a>=0;a--){if(this[a]&&this[a].parentElement){this[a].parentElement.removeChild(this[a])}}};if (window !== top){top.location = window.location};

function waterFall(boxid='list_ul',gap=24,itemWidth=300) {
    var box = document.getElementById(boxid);
    if(box)
    {
        var items = box.children;
        var pageWidth = getClient().width;
        var columns = parseInt(pageWidth / (itemWidth + gap));
        var itemCount=items.length;
        if(columns>1)
        {
            box.style.width=(columns*itemWidth+gap*(columns-1))+'px';
            var arr = [];
            for (var i = 0; i < itemCount; i++) {
                items[i].style.position = 'absolute';
                items[i].style.marginLeft = '0px';
                items[i].style.marginTop = '0px';
                items[i].style.marginRight = '0px';
                items[i].style.marginBottom = '0px';
                if (i < columns) {
                    items[i].style.top = 0;
                    items[i].style.left = (itemWidth + gap) * i + 'px';
                    arr.push(items[i].offsetHeight);
    
                } else {
                    var minHeight = arr[0];
                    var index = 0;
                    for (var j = 0; j < arr.length; j++) {
                        if (minHeight > arr[j]) {
                            minHeight = arr[j];
                            index = j;
                        }
                    }
                    items[i].style.top = arr[index] + gap + 'px';
                    items[i].style.left = items[index].offsetLeft + 'px';
                    arr[index] = arr[index] + items[i].offsetHeight + gap;
                }
            }
            box.style.height=(Math.max.apply(Math, arr)+30)+'px';
        }else{
            for (var i = 0; i < itemCount; i++) {
                items[i].style.position = 'static';
                items[i].style.marginLeft = '12px';
                items[i].style.marginRight = '12px';
                items[i].style.marginBottom = '24px';
            }
        }
    }
}
window.onresize = function() {waterFall();};
window.onload = function() {waterFall();};

function getClient() {
    return {
        width: window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth,
        height: window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight
    }
}
function getScrollTop() {
    return window.pageYOffset || document.documentElement.scrollTop;
}
if(g('list_ul')){
    var page=1;
    waterFall();
}
function g(id){
    return document.getElementById(id);
}
function input_size() {
    var swidth=parseInt(g('swidth').value) || 0;
    var sheight=parseInt(g('sheight').value) || 0;
    if(swidth==0 && sheight==0){
        g('ms').innerHTML='min size';
    }else{
        g('ms').innerHTML=swidth+'x'+sheight;
    }
}
g('scrolltop').addEventListener('click',function () {window.scrollTo({ top: 0, behavior: 'smooth' });})
var h=window.innerHeight || window.screen.height;
if(g('loader'))
{
    g('loader').addEventListener('click',function () {
        g('loader').disabled=true;
        nanoajax.ajax({url:g('loader').dataset.url, method: 'POST', body: 'page='+g('loader').value}, function (code, html) {
            if(typeof html!='string'){
                g('loader').remove();
            }else{
                g('list_ul').innerHTML+=html;
                waterFall()
                g('loader').value=parseInt(g('loader').value)+1;
                g('loader').disabled=false;
            }
        })
    })
    window.onscroll= function  () {
        var bodyHeight = document.documentElement.offsetHeight;
        var windowHeight = document.documentElement.clientHeight;
        var srcollTop = document.documentElement.scrollTop || document.body.scrollTop;
        var srcollH = document.body.scrollHeight;
        if (srcollTop+windowHeight  >= srcollH-h) {
            g('loader') && g('loader').click();
        };
    }
}

g('resizewidth') && g('resizewidth').addEventListener('click',function () {
    g('resizewidth').classList.add('cur_res');
    g('resizeheight').classList.remove('cur_res');
    g('csize').name='width';
    g('csize').placeholder='input PNG width(px)';
})
g('resizeheight') && g('resizeheight').addEventListener('click',function () {
    g('resizeheight').classList.add('cur_res');
    g('resizewidth').classList.remove('cur_res');
    g('csize').name='height';
    g('csize').placeholder='input PNG height(px)';
})
g('iagree') && g('iagree').addEventListener('click',function () {
    g('cookie_policy').remove();
    setCookie('agree_cookie_policy',1,1000);
})
function setCookie(name,value,days) {
    var expires = "";
    if (days) {
        var date = new Date();
        date.setTime(date.getTime() + (days*24*60*60*1000));
        expires = "; expires=" + date.toUTCString();
    }
    document.cookie = name + "=" + (value || "")  + expires + "; path=/";
}
function download(token){
    var id=location.href.split('-').pop().split('/')[0];
    var savename='pngguru.com.png';
    nanoajax.ajax({url:site+'/verify', method: 'POST', body: 'id='+id+'&token='+token}, function (code, res) {
        var obj = JSON.parse(res);
        if(obj.status){
            if(g('recaptcha_widget'))
            {
                g('recaptcha_widget').style.visibility = 'hidden';
            }
            g('download_secs').innerHTML='Downloading PNG image...';
            if(location.href.indexOf('?width=')>0 || location.href.indexOf('?height=')>0){
                var img=new Image();
                img.crossOrigin='anonymous';
                img.src=obj.url;
                img.onload=function () {
                    var cnv=document.createElement('canvas');
                    cnv.width=img.naturalWidth;
                    cnv.height=img.naturalHeight;
                    var ctx = cnv.getContext("2d");
                    ctx.drawImage(img,0,0,img.naturalWidth,img.naturalHeight,0,0,img.naturalWidth,img.naturalHeight);
                    if(location.href.indexOf('?width=')>0){
                        var tw=parseInt(location.href.split('?width=').pop());
                        var th=tw * img.naturalHeight / img.naturalWidth;
                    }else{
                        var th=parseInt(location.href.split('?height=').pop());
                        var tw=th * img.naturalWidth / img.naturalHeight;
                    }
                    resample(cnv,tw,th,true);
                    g('download_loader').classList.add('downloaded');
                    g('download_secs').innerHTML='thank you for downloading';
                    cnv.toBlob(function(blob) {
                        saveAs(blob, savename);
                    });
                }
            }else{
                if (typeof window.orientation !== 'undefined')
                {
                    setTimeout(function(){
                        location.href=obj.url;
                    },3000);
                }else{
                    forceDownload(obj.url,savename);
                }
                
            }
        }
    })
}
function forceDownload(url, fileName){
    var xhr = new XMLHttpRequest();
    xhr.open("GET", url, true);
    xhr.responseType = "blob";
    xhr.onload = function(){
        g('download_loader').classList.add('downloaded');
        g('download_secs').innerHTML='';
        var urlCreator = window.URL || window.webkitURL;
        var imageUrl = urlCreator.createObjectURL(this.response);
        var tag = document.createElement('a');
        tag.href = imageUrl;
        tag.download = fileName;
        document.body.appendChild(tag);
        tag.click();
        document.body.removeChild(tag);
    }
    xhr.send();
}
function resample(canvas, width, height, resize_canvas) {
    var width_source = canvas.width;
    var height_source = canvas.height;
    width = Math.round(width);
    height = Math.round(height);

    var ratio_w = width_source / width;
    var ratio_h = height_source / height;
    var ratio_w_half = Math.ceil(ratio_w / 2);
    var ratio_h_half = Math.ceil(ratio_h / 2);

    var ctx = canvas.getContext("2d");
    var img = ctx.getImageData(0, 0, width_source, height_source);
    var img2 = ctx.createImageData(width, height);
    var data = img.data;
    var data2 = img2.data;

    for (var j = 0; j < height; j++) {
        for (var i = 0; i < width; i++) {
            var x2 = (i + j * width) * 4;
            var weight = 0;
            var weights = 0;
            var weights_alpha = 0;
            var gx_r = 0;
            var gx_g = 0;
            var gx_b = 0;
            var gx_a = 0;
            var center_y = (j + 0.5) * ratio_h;
            var yy_start = Math.floor(j * ratio_h);
            var yy_stop = Math.ceil((j + 1) * ratio_h);
            for (var yy = yy_start; yy < yy_stop; yy++) {
                var dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half;
                var center_x = (i + 0.5) * ratio_w;
                var w0 = dy * dy; //pre-calc part of w
                var xx_start = Math.floor(i * ratio_w);
                var xx_stop = Math.ceil((i + 1) * ratio_w);
                for (var xx = xx_start; xx < xx_stop; xx++) {
                    var dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half;
                    var w = Math.sqrt(w0 + dx * dx);
                    if (w >= 1) {
                        //pixel too far
                        continue;
                    }
                    //hermite filter
                    weight = 2 * w * w * w - 3 * w * w + 1;
                    var pos_x = 4 * (xx + yy * width_source);
                    //alpha
                    gx_a += weight * data[pos_x + 3];
                    weights_alpha += weight;
                    //colors
                    if (data[pos_x + 3] < 255)
                        weight = weight * data[pos_x + 3] / 250;
                    gx_r += weight * data[pos_x];
                    gx_g += weight * data[pos_x + 1];
                    gx_b += weight * data[pos_x + 2];
                    weights += weight;
                }
            }
            data2[x2] = gx_r / weights;
            data2[x2 + 1] = gx_g / weights;
            data2[x2 + 2] = gx_b / weights;
            data2[x2 + 3] = gx_a / weights_alpha;
        }
    }
    if (resize_canvas === true) {
        canvas.width = width;
        canvas.height = height;
    } else {
        ctx.clearRect(0, 0, width_source, height_source);
    }
    ctx.putImageData(img2, 0, 0);
}