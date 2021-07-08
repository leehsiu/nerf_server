var canvas, renderer;
var scene;
var nerfPlot;

var cameraView;
var cameraViewDummy;

var globalView;
var cameraRig;
var nerfImage;

var obj;
var env;
var renderScale = 1.0;
var transformControl;
var startTime;

const point = new THREE.Vector3();
const raycaster = new THREE.Raycaster();

const pointer = new THREE.Vector2();
const onUpPosition = new THREE.Vector2();
const onDownPosition = new THREE.Vector2();

const params = {
    render: nerfRender,
    scale: renderScale
};
class MinMaxGUIHelper {
    constructor(obj, minProp, maxProp, minDif) {
        this.obj = obj;
        this.minProp = minProp;
        this.maxProp = maxProp;
        this.minDif = minDif;
    }
    get min() {
        return this.obj[this.minProp];
    }
    set min(v) {
        this.obj[this.minProp] = v;
        this.obj[this.maxProp] = Math.max(this.obj[this.maxProp], v + this.minDif);
    }
    get max() {
        return this.obj[this.maxProp];
    }
    set max(v) {
        this.obj[this.maxProp] = v;
        this.min = this.min;  // this will call the min setter
    }
}

function updateImage(dataURI){
    var loader = new THREE.TextureLoader();
    loader.load(dataURI,updateTexture);
    var endDate = new Date();
    var timeElapse = (endDate.getTime()-startTime)/1000;
    alert(['Render done in ',timeElapse,'seconds']);
}

function nerfRender() {
    //get camera
    transCamera = cameraView.position;
    rotCamera = cameraView.quaternion;

    //get object information.
    transObj = obj.position;
    rotObj = obj.quaternion;
    scaleObj = obj.scale;
    var bboxObj = new THREE.Box3().setFromObject(obj);

    //customize other controllers here.
    //I recommend dat.gui
    var postData = {
        transCamera:transCamera,
        rotCamera:rotCamera,
        transObj:transObj,
        rotObj:rotObj,
        scaleObj:scaleObj,
        bboxObj:bboxObj,
        renderScale:params.scale
    };

    //Post parameters to python api functions
    startTime = new Date().getTime();
    var request = $.ajax({
        method: "POST",
        url: "api/render",
        data: JSON.stringify(postData)
    });
    request.done(updateImage);
}

function initTexture(texture){
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.NearestFilter;
    //texture.flipY = false;
    var image = texture.image;
    var imageMaterial = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
    var geometry = new THREE.PlaneGeometry(image.width, image.height);

    nerfImage = new THREE.Mesh(geometry, imageMaterial);
    nerfImage.position.x = 0;
    nerfImage.position.y = 0;
    nerfImage.position.z = -1;
    nerfPlot.add(nerfImage);
    render();
}

function updateTexture(texture){
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.NearestFilter;
    var image = texture.image;
    var geometry = new THREE.PlaneGeometry(image.width, image.height);

    nerfImage.material.map.dispose();
    nerfImage.geometry.dispose();
    nerfImage.material.map = texture;
    nerfImage.geometry = geometry;
    nerfImage.position.x = 0;
    nerfImage.position.y = 0;
    nerfImage.position.z = -1;
    render();
}

function init() {
    canvas = $('#gl')[0]
    var aspect = 16 / 9;
    scene = new THREE.Scene();

    //camera
    cameraView = new THREE.PerspectiveCamera(50, aspect, 0.1, 1000);
    cameraViewDummy = new THREE.PerspectiveCamera(50, aspect, 1, 2);
    cameraView.position.set(0, 2, 2);
    cameraViewDummy.position.set(0, 2, 2);
    cameraRig = new THREE.CameraHelper(cameraViewDummy);
    const controlsCameraView = new THREE.OrbitControls(cameraView, $('#camera')[0]);
    controlsCameraView.addEventListener('change', render);
    scene.add(cameraView);
    scene.add(cameraRig);

    //global camera.
    globalView = new THREE.PerspectiveCamera(50, aspect, 1, 1000);
    globalView.position.set(0, 15, 15);
    const controlsGlobalView = new THREE.OrbitControls(globalView, $('#scene')[0]);
    controlsGlobalView.addEventListener('change', render);
    transformControl = new THREE.TransformControls(globalView, $('#scene')[0]);
    transformControl.addEventListener('change', render);

    transformControl.addEventListener('dragging-changed', function (event) {
        controlsGlobalView.enabled = !event.value;
    });

    scene.add(globalView);
    scene.add(transformControl);

    //world grid and light
    scene.add(new THREE.AmbientLight(0xf0f0f0));
    const light = new THREE.SpotLight(0xffffff, 1.5);
    light.position.set(0, 1500, 200);
    light.angle = Math.PI * 0.2;
    light.castShadow = true;
    light.shadow.camera.near = 200;
    light.shadow.camera.far = 2000;
    light.shadow.bias = - 0.000222;
    light.shadow.mapSize.width = 1024;
    light.shadow.mapSize.height = 1024;
    scene.add(light);

    const grid = new THREE.GridHelper(20, 20);
    grid.position.y = -2;
    grid.material.opacity = 0.25;
    grid.material.transparent = true;
    scene.add(grid);

    //The image viewer.
    nerfPlot = new THREE.Scene();
    var width = 4800;
    var height = 900;
    const nerfView = new THREE.OrthographicCamera(width / - 2, width / 2,
        height / 2, height / - 2, 1, 10);
    nerfView.position.set(0, 0, 1);
    nerfPlot.userData.camera = nerfView;
    nerfPlot.add(nerfView);

    const controlsNerf = new THREE.MapControls(nerfView, $('#nerf')[0]);
    controlsNerf.addEventListener('change', render);
    controlsNerf.minDistance = 2;
    controlsNerf.maxDistance = 5;
    controlsNerf.enableRotate = false;


    var loader = new THREE.TextureLoader();
    loader.load(
        'title.png',
        initTexture
    )

    //load the object point and env point cloud.
    //Can be replaced by simple geometry such as a box
    var plyLoader = new THREE.PLYLoader();
    plyLoader.setPropertyNameMapping({
        diffuse_red: 'red',
        diffuse_green: 'green',
        diffuse_blue: 'blue'
    });
    plyLoader.load('ply/hotdog.ply', function (geometry) {
        var ptsMaterial = new THREE.PointsMaterial({ size: 0.005, vertexColors: THREE.VertexColors });
        obj = new THREE.Points(geometry, ptsMaterial);
        scene.add(obj)
    });
    plyLoader.load('ply/fortress.ply', function (geometry) {
        var ptsMaterial = new THREE.PointsMaterial({ size: 0.05, vertexColors: THREE.VertexColors });
        env = new THREE.Points(geometry, ptsMaterial);
        scene.add(env)
    });

    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setClearColor(0xffffff, 1);
    renderer.setPixelRatio(window.devicePixelRatio);

    //dat.GUI() for interactive control
    const gui = new dat.GUI();
    gui.add(params,'render');
    gui.add(params,'scale',0.01,1.0);
    gui.add(cameraView,'fov',1,180).onChange(onCameraUpdate);
    gui.open();


    //start
    document.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('pointerup', onPointerUp);
    document.addEventListener('pointermove', onPointerMove);
    $(document).keydown(onKeyDown);
    transformControl.attach(obj);
    transformControl.setSpace('local');
    render();
}
function onCameraUpdate() {
    cameraView.updateProjectionMatrix();
    cameraRig.update();
    render();
}

function updateSize() {
    $('#scene,#camera').each(function () {
        var view_width = $(this).width();
        var view_height = view_width * 9 / 16;
        $(this).css('height', view_height);
    });
    $('#nerf').css('height', $('#nerf').width() * 9 / 48)
    const width = $('#container')[0].clientWidth;
    const height = $('#container')[0].clientHeight;
    //const height = canvas.clientHeight;
    if (canvas.width !== width || canvas.height !== height) {
        renderer.setSize(width, height, false);
    }
}

function setViewportTo(renderObj,element){
    const rect = element.getBoundingClientRect();
        // check if it's offscreen. If so skip it
    if (rect.bottom < 0 || rect.top > renderObj.domElement.clientHeight ||
        rect.right < 0 || rect.left > renderObj.domElement.clientWidth) {
        return; // it's off screen
    }
    // set the viewport
    const width = rect.right - rect.left;
    const height = rect.bottom - rect.top;
    const left = rect.left;
    const bottom = renderObj.domElement.clientHeight - rect.bottom;
    renderObj.setViewport(left, bottom, width, height);
    renderObj.setScissor(left, bottom, width, height);
}

function updateCameraRig(){
    cameraViewDummy.matrixWorld.copy(cameraView.matrixWorld);
    cameraViewDummy.updateProjectionMatrix();
    cameraRig.update();
}

function render() {
    updateSize();
    updateCameraRig();
    canvas.style.transform = `translateY(${window.scrollY}px)`;
    renderer.setClearColor(0xffffff);
    renderer.setScissorTest(true);

    setViewportTo(renderer,$('#camera')[0]);
    cameraRig.visible = false;
    transformControl.visible = false;
    renderer.render(scene, cameraView);

    setViewportTo(renderer,$('#scene')[0]);
    transformControl.visible = true;
    cameraRig.visible = true;
    renderer.render(scene, globalView);

    setViewportTo(renderer,$('#nerf')[0]);
    renderer.render(nerfPlot, nerfPlot.userData.camera);
}


//keyboard handler
function onKeyDown(event) {
    switch (event.keyCode) {
        case 84: // T
            transformControl.setMode("translate");
            break;

        case 82: // R
            transformControl.setMode("rotate");
            break;

        case 83: // S
            transformControl.setMode("scale");
            break;

        case 88: // X
            transformControl.showX = !transformControl.showX;
            break;

        case 89: // Y
            transformControl.showY = !transformControl.showY;
            break;

        case 90: // Z
            transformControl.showZ = !transformControl.showZ;
            break;
    }
}

function onPointerDown(event) {
    onDownPosition.x = event.clientX;
    onDownPosition.y = event.clientY;

}
function onPointerUp(event) {
    onUpPosition.x = event.clientX;
    onUpPosition.y = event.clientY;
    if (onDownPosition.distanceTo(onUpPosition) === 0) transformControl.detach();
}
function onPointerMove(event) {
    const element = $('#scene')[0];
    const rect = element.getBoundingClientRect();
    const width = rect.right - rect.left;
    const height = rect.bottom - rect.top;
    pointer.x = (event.clientX - rect.left) / width * 2 - 1;
    pointer.y = - (event.clientY - rect.top) / height * 2 + 1;
    raycaster.setFromCamera(pointer, globalView);
    const intersects = raycaster.intersectObjects([obj]);

    //to handle multiple objects.
    if (intersects.length > 0) {
        //const object = intersects[0].object;
        //if (object !== transformControl.object) {
            transformControl.attach(obj);
        //}
    }
    // else{
    //     transformControl.detach(obj);
    // }
}

$(document).ready(init);
$(window).resize(render);