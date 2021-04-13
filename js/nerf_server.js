let canvas, renderer;

const scenes = [];

init();
animate();

function init() {
    canvas = $('#gl')[0]
    // canvas = document.getElementById("gl");
    const geometries = [
        new THREE.BoxGeometry( 1, 1, 1 ),
        new THREE.SphereGeometry( 0.5, 12, 8 ),
        new THREE.DodecahedronGeometry( 0.5 ),
        new THREE.CylinderGeometry( 0.5, 0.5, 1, 12 )
    ];
    //create scene.
    //object,background,camera.
    var scene_frame = $('#scene')[0]
    var scene = new THREE.Scene();
    // make a list item
    const sceneElement = document.createElement('div');
    $(sceneElement).addClass('view');

    scene_frame.appendChild(sceneElement);

    const descriptionElement = document.createElement('div');
    $(descriptionElement).addClass('des');
    descriptionElement.innerText = 'Scene';
    scene_frame.appendChild(descriptionElement);
    // the element that represents the area we want to render the scene
    scene.userData.element = sceneElement;

    var aspect = 16/9;
    const camera = new THREE.PerspectiveCamera(50, aspect, 1, 10);
    camera.position.z = 2;
    scene.userData.camera = camera;

    const controls = new THREE.OrbitControls(scene.userData.camera, scene.userData.element);
    controls.minDistance = 2;
    controls.maxDistance = 5;
    controls.enablePan = false;
    controls.enableZoom = false;
    scene.userData.controls = controls;

    // add one random mesh to each scene
    const geometry = geometries[geometries.length * Math.random() | 0];

    const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color().setHSL(Math.random(), 1, 0.75),
        roughness: 0.5,
        metalness: 0,
        flatShading: true

    });

    scene.add(new THREE.Mesh(geometry, material));

    scene.add(new THREE.HemisphereLight(0xaaaaaa, 0x444444));

    const light = new THREE.DirectionalLight(0xffffff, 0.5);
    light.position.set(1, 1, 1);
    scene.add(light);

    scenes.push(scene);

    //create camera.
    //put into local view, but without camera

    var camera_frame = $('#camera')[0]
    {
        const scene = new THREE.Scene();

        const sceneElement = document.createElement('div');
        camera_frame.appendChild(sceneElement);
        $(sceneElement).addClass('view');
        const descriptionElement = document.createElement('div');
        descriptionElement.innerText = 'Camera View';
        $(descriptionElement).addClass('des')
        camera_frame.appendChild(descriptionElement);
        // the element that represents the area we want to render the scene
        scene.userData.element = sceneElement;

        const camera = new THREE.PerspectiveCamera(50, 16/9, 1, 10);
        camera.position.z = 2;
        scene.userData.camera = camera;

        const controls = new THREE.OrbitControls(scene.userData.camera, scene.userData.element);
        controls.minDistance = 2;
        controls.maxDistance = 5;
        controls.enablePan = false;
        controls.enableZoom = false;
        scene.userData.controls = controls;

        // add one random mesh to each scene
        const geometry = geometries[geometries.length * Math.random() | 0];

        const material = new THREE.MeshStandardMaterial({

            color: new THREE.Color().setHSL(Math.random(), 1, 0.75),
            roughness: 0.5,
            metalness: 0,
            flatShading: true

        });

        scene.add(new THREE.Mesh(geometry, material));

        scene.add(new THREE.HemisphereLight(0xaaaaaa, 0x444444));

        const light = new THREE.DirectionalLight(0xffffff, 0.5);
        light.position.set(1, 1, 1);
        scene.add(light);

        scenes.push(scene);

    }
    //NeRF rendered
    var nerf_frame = $('#nerf')[0]
    {
        const scene = new THREE.Scene();

        const sceneElement = document.createElement('div');
        nerf_frame.appendChild(sceneElement);
        $(sceneElement).addClass('view');
        const descriptionElement = document.createElement('div');
        descriptionElement.innerText = 'NeRF rendered';
        $(descriptionElement).addClass('des')
        nerf_frame.appendChild(descriptionElement);
        // the element that represents the area we want to render the scene
        scene.userData.element = sceneElement;

        
        var width = 1000;
        var height = 1000;
        const camera = new THREE.OrthographicCamera(width / - 2, width / 2, 
        height / 2, height / - 2, 1, 10);
        camera.position.set(0, 0, 1);
        scene.userData.camera = camera;

        const controls = new THREE.OrbitControls(scene.userData.camera, scene.userData.element);
        controls.minDistance = 2;
        controls.maxDistance = 5;
        controls.enableRotate = false;
        scene.userData.controls = controls;

        var loader = new THREE.TextureLoader();
        loader.crossOrigin = '';
        const texture_plane = loader.load('http://i.imgur.com/3tU4Vig.jpg');

        // plane to display
        const geometry_plane = new THREE.PlaneGeometry(512, 512, 1);
        const mesh_plane = new THREE.Mesh(geometry_plane,
            new THREE.MeshLambertMaterial({ map: texture_plane }))
        scene.add(mesh_plane);

        // just camera and light
        scene.add(camera);
        const light = new THREE.AmbientLight( 0xffffff );
        scene.add(light);
        scenes.push(scene);

    }
    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setClearColor(0xffffff, 1);
    renderer.setPixelRatio(window.devicePixelRatio);

}

function updateSize() {
    $('.view').each(function(){
        var view_width = $(this).width();
        var view_height = view_width*9/16;
        $(this).css('height',view_height);
    });


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

    scenes.forEach(function (scene) {
    
        // get the element that is a place holder for where we want to
        // draw the scene
        const element = scene.userData.element;

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

        const camera = scene.userData.camera;

        //camera.aspect = width / height; // not changing in this example
        //camera.updateProjectionMatrix();
        // console.log(scene.userData.controls)
        //scene.userData.controls.update();

        renderer.render(scene, camera);

    });

}
