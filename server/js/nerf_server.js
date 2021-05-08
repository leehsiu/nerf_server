var canvas, renderer;
var scene;
var nerf_plot;

var camera_view;
var global_view;
var camera_rig;
var nerf_image;

var obj;
var env;
var transformControl;

const point = new THREE.Vector3();
const raycaster = new THREE.Raycaster();

const pointer = new THREE.Vector2();
const onUpPosition = new THREE.Vector2();
const onDownPosition = new THREE.Vector2();
const params = {
    render: NeRF_render,
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



function updateImage(data) {
    alert('get reply');
    console.log(data);
    
}

function NeRF_render() {
    //get camera
    trans = obj.position;
    rot = obj.quaternion;
    scale = obj.scale;
    //x,y,z.
    var bbox = new THREE.Box3().setFromObject(obj);
    var request = $.ajax({
        method: "POST",
        url: "api/render",
        data: JSON.stringify({trans:trans,rotation:rot,scale:scale,bbox:bbox})
    });
    request.done(updateImage);
}

//Init function. Parsing parameters.

function init() {
    canvas = $('#gl')[0]
    var aspect = 16 / 9;
    scene = new THREE.Scene();

    //camera
    camera_view = new THREE.PerspectiveCamera(50, aspect, 1, 50);
    camera_view.position.set(0, 2, 2);
    camera_rig = new THREE.CameraHelper(camera_view);
    const controls_camera_view = new THREE.OrbitControls(camera_view, $('#camera')[0]);
    controls_camera_view.addEventListener('change', render);
    scene.add(camera_view);
    scene.add(camera_rig);

    //global camera.
    global_view = new THREE.PerspectiveCamera(50, aspect, 1, 1000);
    global_view.position.set(0, 15, 15);
    const controls_global_view = new THREE.OrbitControls(global_view, $('#scene')[0]);
    controls_global_view.addEventListener('change', render);
    transformControl = new THREE.TransformControls(global_view, $('#scene')[0]);
    transformControl.addEventListener('change', render);

    transformControl.addEventListener('dragging-changed', function (event) {
        controls_global_view.enabled = !event.value;
    });

    scene.add(global_view);
    scene.add(transformControl);


    //create grid and light
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


    //create the image viewer.
    nerf_plot = new THREE.Scene();
    var width = 4800;
    var height = 900;
    const nerf_view = new THREE.OrthographicCamera(width / - 2, width / 2,
        height / 2, height / - 2, 1, 10);
    nerf_view.position.set(0, 0, 1);
    nerf_plot.userData.camera = nerf_view;
    nerf_plot.add(nerf_view);

    const controls_nerf = new THREE.MapControls(nerf_view, $('#nerf')[0]);
    controls_nerf.addEventListener('change', render);
    controls_nerf.minDistance = 2;
    controls_nerf.maxDistance = 5;
    controls_nerf.enableRotate = false;


    var loader = new THREE.TextureLoader();
    loader.crossOrigin = '';
    const texture_plane = loader.load('img.png');

    //plane to display nerf_element
    const geometry_plane = new THREE.PlaneGeometry(1920, 360, 1);
    nerf_image = new THREE.Mesh(geometry_plane,
        new THREE.MeshLambertMaterial({ map: texture_plane }))
    nerf_plot.add(nerf_image);
    nerf_plot.add(new THREE.AmbientLight(0xffffff));

    var ply_loader = new THREE.PLYLoader();
    ply_loader.setPropertyNameMapping({
        diffuse_red: 'red',
        diffuse_green: 'green',
        diffuse_blue: 'blue'
    });
    ply_loader.load('ply/hotdog.ply', function (geometry) {
        var pts_material = new THREE.PointsMaterial({ size: 0.005, vertexColors: THREE.VertexColors });
        obj = new THREE.Points(geometry, pts_material);
        scene.add(obj)
    });
    //bbox = new THREE.Box3().setFromObject(obj);
    ply_loader.load('ply/fortress.ply', function (geometry) {
        var pts_material = new THREE.PointsMaterial({ size: 0.05, vertexColors: THREE.VertexColors });
        env = new THREE.Points(geometry, pts_material);
        scene.add(env)
    });


    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setClearColor(0xffffff, 1);
    renderer.setPixelRatio(window.devicePixelRatio);

    const gui = new dat.GUI();
    gui.add(params, 'render');
    gui.add(camera_view, 'fov', 1, 180).onChange(onCameraUpdate);
    const minMaxGUIHelper = new MinMaxGUIHelper(camera_view, 'near', 'far', 0.1);
    gui.add(minMaxGUIHelper, 'min', 0.1, 50, 0.1).name('near').onChange(onCameraUpdate);
    gui.add(minMaxGUIHelper, 'max', 0.1, 50, 0.1).name('far').onChange(onCameraUpdate);
    gui.open();


    //Controls
    document.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('pointerup', onPointerUp);
    document.addEventListener('pointermove', onPointerMove);
    $(document).keydown(onKeyDown);
    transformControl.attach(obj);
    transformControl.setSpace('local');
    render();
}
function onCameraUpdate() {
    camera_view.updateProjectionMatrix();
    camera_rig.update();
    render();
}
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

function render() {

    updateSize();
    canvas.style.transform = `translateY(${window.scrollY}px)`;

    renderer.setClearColor(0xffffff);
    renderer.setScissorTest(true);

    {
        // get the element that is a place holder for where we want to
        // draw the scene
        const element = $('#camera')[0];
        // get its position relative to the page's viewport
        const rect = element.getBoundingClientRect();

        // check if it's offscreen. If so skip it
        if (rect.bottom < 0 || rect.top > renderer.domElement.clientHeight ||
            rect.right < 0 || rect.left > renderer.domElement.clientWidth) {
            return; // it's off screen
        }

        // set the viewport
        const width = rect.right - rect.left;
        const height = rect.bottom - rect.top;
        const left = rect.left;
        const bottom = renderer.domElement.clientHeight - rect.bottom;

        renderer.setViewport(left, bottom, width, height);
        renderer.setScissor(left, bottom, width, height);

        camera_rig.visible = false;
        renderer.render(scene, camera_view);
    }
    {
        // get the element that is a place holder for where we want to
        // draw the scene
        const element = $('#scene')[0];
        // get its position relative to the page's viewport
        const rect = element.getBoundingClientRect();

        // check if it's offscreen. If so skip it
        if (rect.bottom < 0 || rect.top > renderer.domElement.clientHeight ||
            rect.right < 0 || rect.left > renderer.domElement.clientWidth) {
            return; // it's off screen
        }

        // set the viewport
        const width = rect.right - rect.left;
        const height = rect.bottom - rect.top;
        const left = rect.left;
        const bottom = renderer.domElement.clientHeight - rect.bottom;
        renderer.setViewport(left, bottom, width, height);
        renderer.setScissor(left, bottom, width, height);
        camera_rig.visible = true;
        renderer.render(scene, global_view);
    }
    {
        // get the element that is a place holder for where we want to
        // draw the scene
        const element = $('#nerf')[0];
        // get its position relative to the page's viewport
        const rect = element.getBoundingClientRect();

        // check if it's offscreen. If so skip it
        if (rect.bottom < 0 || rect.top > renderer.domElement.clientHeight ||
            rect.right < 0 || rect.left > renderer.domElement.clientWidth) {
            return; // it's off screen
        }
        // set the viewport
        const width = rect.right - rect.left;
        const height = rect.bottom - rect.top;
        const left = rect.left;
        const bottom = renderer.domElement.clientHeight - rect.bottom;
        renderer.setViewport(left, bottom, width, height);
        renderer.setScissor(left, bottom, width, height);
        renderer.render(nerf_plot, nerf_plot.userData.camera)
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
    raycaster.setFromCamera(pointer, global_view);
    const intersects = raycaster.intersectObjects([obj]);
    if (intersects.length > 0) {
        const object = intersects[0].object;
        if (object !== transformControl.object) {
            transformControl.attach(object);
        }
    }
    // else{
    //     transformControl.detach(obj);
    // }
}

$(window).resize(render);
$(document).ready(init);