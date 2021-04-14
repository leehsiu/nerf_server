var canvas, renderer;
//
var scene;
var nerf;
var local_camera;
var global_camera;
var camera_rig;

const geometries = [
    new THREE.BoxGeometry( 1, 1, 1 ),
    new THREE.SphereGeometry( 0.5, 12, 8 ),
    new THREE.DodecahedronGeometry( 0.5 ),
    new THREE.CylinderGeometry( 0.5, 0.5, 1, 12 )
];

//for spline.
const splineHelperObjects = [];
const positions = [];

const point = new THREE.Vector3();
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const onUpPosition = new THREE.Vector2();
const onDownPosition = new THREE.Vector2();
let transformControl;
const ARC_SEGMENTS = 200;
const splines = {};
const params = {
	addPoint: addPoint,
	removePoint: removePoint,
    render:NeRF_render,
};

function NeRF_render(){

}



init();
animate();

function init() {
    canvas = $('#gl')[0]

    var aspect = 16/9;
    scene = new THREE.Scene();
    local_camera = new THREE.PerspectiveCamera(50,aspect,1,1000);
    local_camera.position.set(0,2,2);
    camera_rig = new THREE.CameraHelper(local_camera);
    global_camera = new THREE.PerspectiveCamera(50,aspect,1,1000);
    global_camera.position.set( 0, 15,15);
    const controls_local = new THREE.OrbitControls(local_camera, $('#camera')[0]);

    scene.add(camera_rig);

    scene.add( new THREE.AmbientLight( 0xf0f0f0 ) );
    const light = new THREE.SpotLight( 0xffffff, 1.5 );
    light.position.set( 0, 1500, 200 );
    light.angle = Math.PI * 0.2;
    light.castShadow = true;
    light.shadow.camera.near = 200;
    light.shadow.camera.far = 2000;
    light.shadow.bias = - 0.000222;
    light.shadow.mapSize.width = 1024;
    light.shadow.mapSize.height = 1024;
    scene.add( light );

    const grid = new THREE.GridHelper(20,20);
    grid.position.y = -2;
    grid.material.opacity = 0.25;
    grid.material.transparent = true;
    scene.add(grid);

    //create the image viewer.
    nerf = new THREE.Scene();
    var width = 4800;
    var height = 900;
    const camera = new THREE.OrthographicCamera(width / - 2, width / 2, 
    height / 2, height / - 2, 1, 10);
    camera.position.set(0, 0, 1);

    const controls3 = new THREE.OrbitControls(camera, $('#nerf')[0]);
    controls3.minDistance = 2;
    controls3.maxDistance = 5;
    controls3.enableRotate = false;

    var loader = new THREE.TextureLoader();
    loader.crossOrigin = '';
    const texture_plane = loader.load('http://i.imgur.com/3tU4Vig.jpg');

    // plane to display
    const geometry_plane = new THREE.PlaneGeometry(512, 512, 1);
    const mesh_plane = new THREE.Mesh(geometry_plane,
        new THREE.MeshLambertMaterial({ map: texture_plane }))
    nerf.add(mesh_plane);

    nerf.userData.camera = camera;
    // just camera and light
    nerf.add(camera);{
        const light = new THREE.AmbientLight( 0xffffff );
        nerf.add(light);
    }

    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setClearColor(0xffffff, 1);
    renderer.setPixelRatio(window.devicePixelRatio);

    const gui = new dat.GUI();
	gui.add(params, 'addPoint');
	gui.add(params, 'removePoint');
    gui.add(params,'render')
	gui.open();

	// Controls
    const controls = new THREE.OrbitControls(global_camera,$('#scene')[0]);
    controls.damping = 0.2;
    controls.addEventListener( 'change', render );
    transformControl = new THREE.TransformControls( global_camera, $('#scene')[0]);
    transformControl.addEventListener( 'change', render );
    transformControl.addEventListener( 'dragging-changed', function ( event ) {
        controls.enabled = ! event.value;
    } );
    scene.add( transformControl );

    transformControl.addEventListener( 'objectChange', function () {
        updateSplineOutline();
    } );

	document.addEventListener('pointerdown',onPointerDown);
	document.addEventListener('pointerup',onPointerUp);
	document.addEventListener('pointermove',onPointerMove);

	positions.length = 0;
}

function updateSize() {
    $('#scene,#camera').each(function(){
        var view_width = $(this).width();
        var view_height = view_width*9/16;
        $(this).css('height',view_height);
    });
    $('#nerf').css('height',$('#nerf').width()*9/48)

    const width = $('#container')[0].clientWidth;
    const height = $('#container')[0].clientHeight;
    //const height = canvas.clientHeight;
    if (canvas.width !== width || canvas.height !== height) {
        renderer.setSize(width, height, false);
    }

}

function animate() {

    render();
    requestAnimationFrame(animate);

}

function render() {

    updateSize();

    canvas.style.transform = `translateY(${window.scrollY}px)`;

    renderer.setClearColor(0xffffff);
    renderer.setScissorTest(false);
    renderer.clear();

    renderer.setClearColor(0xe0e0e0);
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
        // console.log(rect)

        renderer.setViewport(left, bottom, width, height);
        renderer.setScissor(left, bottom, width, height);

        camera_rig.visible = false;

        //camera.aspect = width / height; // not changing in this example
        //camera.updateProjectionMatrix();
        // console.log(scene.userData.controls)
        //scene.userData.controls.update();

        renderer.render(scene, local_camera);
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
        renderer.render(scene, global_camera);
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
        renderer.render(nerf,nerf.userData.camera)
    }

}



function addSplineObject(position) {

    const geometry = geometries[geometries.length * Math.random() | 0]
	const material = new THREE.MeshLambertMaterial({ color: Math.random() * 0xffffff });
	const object = new THREE.Mesh(geometry, material);
	if (position) {
		object.position.copy(position);
	} else {
		object.position.x = Math.random() * 10 - 5;
		object.position.y = Math.random() * 6;
		object.position.z = Math.random() * 8 - 4;

	}

	object.castShadow = true;
	object.receiveShadow = true;
	scene.add(object);
	splineHelperObjects.push(object);
	return object;

}
function addPoint() {
	positions.push(addSplineObject().position);
	updateSplineOutline();

}

function removePoint() {

	const point = splineHelperObjects.pop();
	positions.pop();

	if (transformControl.object === point) transformControl.detach();
	scene.remove(point);

	updateSplineOutline();

}

function updateSplineOutline() {

	for (const k in splines) {

		const spline = splines[k];

		const splineMesh = spline.mesh;
		const position = splineMesh.geometry.attributes.position;

		for (let i = 0; i < ARC_SEGMENTS; i++) {

			const t = i / (ARC_SEGMENTS - 1);
			spline.getPoint(t, point);
			position.setXYZ(i, point.x, point.y, point.z);

		}

		position.needsUpdate = true;

	}

}


function load(new_positions) {

	while (new_positions.length > positions.length) {

		addPoint();

	}

	while (new_positions.length < positions.length) {

		removePoint();

	}

	for (let i = 0; i < positions.length; i++) {

		positions[i].copy(new_positions[i]);

	}

	updateSplineOutline();

}

function onPointerDown(event) {

	onDownPosition.x = event.clientX;
	onDownPosition.y = event.clientY;

}

function onPointerUp() {
	onUpPosition.x = event.clientX;
	onUpPosition.y = event.clientY;
	if (onDownPosition.distanceTo(onUpPosition) === 0) transformControl.detach();

}
function onPointerMove(event) {

    const element = $('#scene')[0];
    const rect = element.getBoundingClientRect();
    const width = rect.right - rect.left;
    const height = rect.bottom - rect.top;
	pointer.x = (event.clientX-rect.left)/width * 2 - 1;
	pointer.y = - (event.clientY-rect.top)/height * 2 + 1;
	raycaster.setFromCamera(pointer, global_camera);
	const intersects = raycaster.intersectObjects(splineHelperObjects);
	if (intersects.length > 0) {
		const object = intersects[0].object;
		if (object !== transformControl.object) {
			transformControl.attach(object);
		}

	}

}