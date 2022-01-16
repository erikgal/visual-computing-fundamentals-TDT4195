extern crate nalgebra_glm as glm;
use std::{ mem, ptr, os::raw::c_void };
use std::thread;
use std::sync::{Mutex, Arc, RwLock};
use toolbox::simple_heading_animation;

mod shader;
mod util;
mod mesh;
mod scene_graph;
mod toolbox;

use glutin::event::{Event, WindowEvent, DeviceEvent, KeyboardInput, ElementState::{Pressed, Released}, VirtualKeyCode::{self, *}};
use glutin::event_loop::ControlFlow;
use scene_graph::SceneNode;
use toolbox::Heading;

const SCREEN_W: u32 = 1600;
const SCREEN_H: u32 = 1100;

// == // Helper functions to make interacting with OpenGL a little bit prettier. You *WILL* need these! // == //
// The names should be pretty self explanatory
fn byte_size_of_array<T>(val: &[T]) -> isize {
    std::mem::size_of_val(&val[..]) as isize
}

// Get the OpenGL-compatible pointer to an arbitrary array of numbers
fn pointer_to_array<T>(val: &[T]) -> *const c_void {
    &val[0] as *const T as *const c_void
}

// Get the size of the given type in bytes
fn size_of<T>() -> i32 {
    mem::size_of::<T>() as i32
}

// Get an offset in bytes for n units of type T
fn offset<T>(n: u32) -> *const c_void {
    (n * mem::size_of::<T>() as u32) as *const T as *const c_void
}

unsafe fn create_vao(vertex_coordinates: &Vec<f32>,  array_of_indices: &Vec<u32>, color_vector: &Vec<f32>, normal_vector: &Vec<f32>) -> u32 {
    // Create and bind VAO
    let mut vao: gl::types::GLuint = 0;
    gl::GenVertexArrays(1, &mut vao);
    gl::BindVertexArray(vao);

    // Create and bind VBO
    let mut vbo: gl::types::GLuint = 0;
    gl::GenBuffers(1, &mut vbo);
    gl::BindBuffer(gl::ARRAY_BUFFER, vbo);

    //Fill VBO
    gl::BufferData(
        gl::ARRAY_BUFFER, // target
        byte_size_of_array(vertex_coordinates), // size
        pointer_to_array(vertex_coordinates), // pointer to data
        gl::STATIC_DRAW, // usage
    );

    gl::VertexAttribPointer(
        0, // ("layout (location = 0)")
        3, // the number of components per generic vertex attribute
        gl::FLOAT, // data type
        gl::FALSE, // normalized 
        size_of::<f32>() * 3, // stride
        offset::<f32>(0) // offset of the first component
    );
    gl::EnableVertexAttribArray(0);

    // Create and bind color VBO
    let mut color_vbo: gl::types::GLuint = 1;
    gl::GenBuffers(1, &mut color_vbo);
    gl::BindBuffer(gl::ARRAY_BUFFER, color_vbo);

    //Fill Color VBO 
    gl::BufferData(
        gl::ARRAY_BUFFER, // target
        byte_size_of_array(color_vector), // size
        pointer_to_array(color_vector), // pointer to data
        gl::STATIC_DRAW, // usage
    );

    gl::VertexAttribPointer(
        1, // ("layout (location = 1)")
        4, // the number of components per generic vertex attribute
        gl::FLOAT, // data type
        gl::FALSE, // normalized 
        size_of::<f32>() * 4, // stride
        offset::<f32>(0), // offset of the first component
    );
    gl::EnableVertexAttribArray(1);

     // Create and bind normal VBO
     let mut normal_vbo: gl::types::GLuint = 2;
     gl::GenBuffers(1, &mut normal_vbo);
     gl::BindBuffer(gl::ARRAY_BUFFER, normal_vbo);

    //Fill normal VBO
    gl::BufferData(
        gl::ARRAY_BUFFER, // target
        byte_size_of_array(normal_vector), // size
        pointer_to_array(normal_vector), // pointer to data
        gl::STATIC_DRAW, // usage
    );

    gl::VertexAttribPointer(
        2, // ("layout (location = 2)")
        3, // the number of components per generic vertex attribute
        gl::FLOAT, // data type
        gl::FALSE, // normalized 
        size_of::<f32>() * 3, // stride
        offset::<f32>(0), // offset of the first component
    );
    gl::EnableVertexAttribArray(2);

    let mut buffer: gl::types::GLuint = 0;
    gl::GenBuffers(1, &mut buffer);
    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, buffer);

     //Fill index buffer
     gl::BufferData(
        gl::ELEMENT_ARRAY_BUFFER, // target
        byte_size_of_array(array_of_indices), // size of data in bytes
        pointer_to_array(array_of_indices), // pointer to data
        gl::STATIC_DRAW, // usage
    );

    return vao
}

unsafe fn draw_scene(node: &scene_graph::SceneNode,
    view_projection_matrix: &glm::Mat4, shaders: &shader::Shader) {
    // Check if node is drawable, set uniforms, draw
    if node.vao_id != 0{
        // print!("{:?}", node.vao_id);

        // Setup uniform
        let mvp = view_projection_matrix * node.current_transformation_matrix;
        let name = String::from("transformation");
        let transformation_location: gl::types::GLint = shaders.get_uniform_location(&name);
        gl::UniformMatrix4fv(transformation_location, 1, gl::FALSE, mvp.as_ptr());

        let name = String::from("model");
        let rotation_location: gl::types::GLint = shaders.get_uniform_location(&name);
        gl::UniformMatrix4fv(rotation_location, 1, gl::FALSE, node.current_transformation_matrix.as_ptr());

        // Call draw
        gl::BindVertexArray(node.vao_id);
        gl::DrawElements(gl::TRIANGLES, node.index_count, gl::UNSIGNED_INT, ptr::null()); 
    }
    // Recurse
    for &child in &node.children {
        draw_scene(&*child, view_projection_matrix, shaders);
    }
}

unsafe fn update_node_transformations(node: &mut scene_graph::SceneNode,
transformation_so_far: &glm::Mat4, elapsed_time: f32,  shaders: &shader::Shader) {
    // Construct the correct transformation matrix
    let mut position: glm::Mat4 =  glm::translation(&node.position);
    let reference: glm::Mat4 =  glm::translation(&node.reference_point);
    let mut rotation: glm::Mat4 = glm::identity();

    // Animated helicopter body path
    if node.vao_id == 2 {
        let heading: Heading = simple_heading_animation(elapsed_time);
        position =  glm::translation(&glm::vec3(heading.x, node.position[1], heading.z));
        let x_rotation  = glm::rotation(heading.pitch, &glm::vec3(1.0, 0.0, 0.0));
        let y_rotation  = glm::rotation(heading.yaw, &glm::vec3(0.0, 1.0, 0.0));
        let z_rotation  = glm::rotation(heading.roll, &glm::vec3(0.0, 0.0, 1.0));
        let xyz_rotation = x_rotation * y_rotation * z_rotation; 
        rotation =  reference * xyz_rotation * glm::inverse(&reference); 
    }

    if vec![3, 4].contains(&node.vao_id) {
        rotation =  reference * glm::rotation(elapsed_time*50.0, &node.rotation) * glm::inverse(&reference); 
    }

    let current_transformation_matrix:  glm::Mat4 = transformation_so_far *  position * rotation; 
    
    // Update the node's transformation matrix
    node.current_transformation_matrix = current_transformation_matrix;

    // Recurse
    for &child in &node.children {
    update_node_transformations(&mut *child,
    &node.current_transformation_matrix, elapsed_time, shaders);
    }
}

fn main() {
    // Set up the necessary objects to deal with windows and event handling
    let el = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_title("Gloom-rs")
        .with_resizable(false)
        .with_inner_size(glutin::dpi::LogicalSize::new(SCREEN_W, SCREEN_H));
    let cb = glutin::ContextBuilder::new()
        .with_vsync(true);
    let windowed_context = cb.build_windowed(wb, &el).unwrap();
    // Uncomment these if you want to use the mouse for controls, but want it to be confined to the screen and/or invisible.
    // windowed_context.window().set_cursor_grab(true).expect("failed to grab cursor");
    // windowed_context.window().set_cursor_visible(false);

    // Set up a shared vector for keeping track of currently pressed keys
    let arc_pressed_keys = Arc::new(Mutex::new(Vec::<VirtualKeyCode>::with_capacity(10)));
    // Make a reference of this vector to send to the render thread
    let pressed_keys = Arc::clone(&arc_pressed_keys);

    // Set up shared tuple for tracking mouse movement between frames
    let arc_mouse_delta = Arc::new(Mutex::new((0f32, 0f32)));
    // Make a reference of this tuple to send to the render thread
    let mouse_delta = Arc::clone(&arc_mouse_delta);

    // Spawn a separate thread for rendering, so event handling doesn't block rendering
    let render_thread = thread::spawn(move || {
        // Acquire the OpenGL Context and load the function pointers. This has to be done inside of the rendering thread, because
        // an active OpenGL context cannot safely traverse a thread boundary
        let context = unsafe {
            let c = windowed_context.make_current().unwrap();
            gl::load_with(|symbol| c.get_proc_address(symbol) as *const _);
            c
        };

        // Set up openGL
        unsafe {
            gl::Enable(gl::DEPTH_TEST);
            gl::DepthFunc(gl::LESS);
            gl::Enable(gl::CULL_FACE);
            gl::Disable(gl::MULTISAMPLE);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
            gl::DebugMessageCallback(Some(util::debug_callback), ptr::null());

            // Print some diagnostics
            println!("{}: {}", util::get_gl_string(gl::VENDOR), util::get_gl_string(gl::RENDERER));
            println!("OpenGL\t: {}", util::get_gl_string(gl::VERSION));
            println!("GLSL\t: {}", util::get_gl_string(gl::SHADING_LANGUAGE_VERSION));
        }

        // Set up VAO
        let mut vaos: Vec<u32> = Vec::new();
        let mut meshes: Vec<mesh::Mesh> = Vec::new();
        let lunarsurface = mesh::Terrain::load("./lunarsurface.obj");
        let helicopter = mesh::Helicopter::load("./helicopter.obj");
        unsafe {
            vaos.push(create_vao(&lunarsurface.vertices , &lunarsurface.indices, &lunarsurface.colors, &lunarsurface.normals));
            meshes.push(lunarsurface);
            for i in 1..5 {
                let helicopter_mesh = &helicopter[i-1];
                vaos.push(create_vao(&helicopter_mesh.vertices, &helicopter_mesh.indices, &helicopter_mesh.colors, &helicopter_mesh.normals));
                meshes.push(helicopter[i-1].clone());
            }
        }

        //Set up scene graph//

        // Create scene nodes
        let mut root = SceneNode::new();
        let mut lunarsurface_node = SceneNode::from_vao(vaos[0], meshes[0].index_count);
        let mut helicopter_body_node = SceneNode::from_vao(vaos[1], meshes[1].index_count);
        let mut helicopter_mainrotor_node = SceneNode::from_vao(vaos[2], meshes[2].index_count);
        let mut helicopter_tailrotor_node = SceneNode::from_vao(vaos[3], meshes[3].index_count);
        let mut helicopter_door_node = SceneNode::from_vao(vaos[4], meshes[4].index_count);


        // Initialise node values
        lunarsurface_node.position = glm::vec3(0.0, 0.0, 0.0);

        helicopter_body_node.position = glm::vec3(0.0, 0.0, 0.0);

        helicopter_mainrotor_node.position = glm::vec3(0.0, 0.0, 0.0);
        helicopter_mainrotor_node.rotation = glm::vec3(0.0, 1.0, 0.0);
        helicopter_mainrotor_node.reference_point = glm::vec3(0.0, 0.0, 0.0); // Rotate about global y-axis, already aligned => no need to move it to origin

        helicopter_tailrotor_node.position = glm::vec3(0.0, 0.0, 0.0);
        helicopter_tailrotor_node.rotation = glm::vec3(1.0, 0.0, 0.0);
        helicopter_tailrotor_node.reference_point = glm::vec3(0.35, 2.1, 9.45); // Move to origin to rotate about x-axis

        helicopter_door_node.position = glm::vec3(0.0, 0.0, 0.0);
        helicopter_door_node.rotation = glm::vec3(0.0, 0.0, 0.0);

        //Build scene graph
        root.add_child(&lunarsurface_node);
        lunarsurface_node.add_child(&helicopter_body_node);
        helicopter_body_node.add_child(&helicopter_door_node);
        helicopter_body_node.add_child(&helicopter_mainrotor_node);
        helicopter_body_node.add_child(&helicopter_tailrotor_node);

        // Set up shaders
        let shaders: shader::Shader;
        unsafe { shaders = shader::ShaderBuilder::new()
            .attach_file("./shaders/simple.frag")
            .attach_file("./shaders/simple.vert")
            .link();
            shaders.activate();
        };

        // Used to demonstrate keyboard handling -- feel free to remove
        let mut movement_coordinates= glm::vec3(111.72193, -11.9816475, -2.918501);
        let mut rotation_coordinates = glm::vec3(1.5564828, -0.01546875, 0.0);

        let movement_const : f32 = 30.0;
        let rotation_const : f32 = 1.5;

        let first_frame_time = std::time::Instant::now();
        let mut last_frame_time = first_frame_time;
        // The main rendering loop
        loop {
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(first_frame_time).as_secs_f32();
            let delta_time = now.duration_since(last_frame_time).as_secs_f32();
            last_frame_time = now;

            // print!("mov {:?}", movement_coordinates);
            // print!("rot {:?}", rotation_coordinates);

            // Handle keyboard input
            // Movement                              Rotation
            //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            //$ Direction $   X   $   Y   $   Z   $  $ Direction $   Yaw     $  Pitch  $
            //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            //$ Forwards  $   A   $   E   $   W   $  $ Forwards  $   Right   $   Up    $
            //$ Backwards $   D   $   Q   $   S   $  $ Backwards $   Left    $   Down  $
            //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            if let Ok(keys) = pressed_keys.lock() {
                for key in keys.iter() {
                    match key {
                        VirtualKeyCode::A => {
                            movement_coordinates[0] += delta_time * movement_const;
                        },
                        VirtualKeyCode::D => {
                            movement_coordinates[0] -= delta_time * movement_const;
                        },
                        VirtualKeyCode::E => {
                            movement_coordinates[1] -= delta_time * movement_const;
                        },
                        VirtualKeyCode::Q => {
                            movement_coordinates[1] += delta_time * movement_const;
                        },
                        VirtualKeyCode::W => {
                            movement_coordinates[2] += delta_time * movement_const;
                        },
                        VirtualKeyCode::S => {
                            movement_coordinates[2] -= delta_time * movement_const;
                        },
                        VirtualKeyCode::Right => {
                            rotation_coordinates[0] += delta_time * rotation_const;
                        },
                        VirtualKeyCode::Left => {
                            rotation_coordinates[0] -= delta_time * rotation_const;
                        },
                        VirtualKeyCode::Up => {
                            rotation_coordinates[1] += delta_time * rotation_const;
                        },
                        VirtualKeyCode::Down => {
                            rotation_coordinates[1] -= delta_time * rotation_const;
                        },
                        _ => { }
                    }
                }
            }
            // Handle mouse movement. delta contains the x and y movement of the mouse since last frame in pixels
            if let Ok(mut delta) = mouse_delta.lock() {
                *delta = (0.0, 0.0);
            }
            unsafe {
                // Uniform scale based on time elapsed
                let name = String::from("scale");
                let scale_location: gl::types::GLint = shaders.get_uniform_location(&name);
                gl::Uniform1f(scale_location, elapsed.sin());

                // Model
                let model: glm::Mat4 = glm::translation(&glm::vec3(0.0, 0.0, 0.0));

                // View
                let xyz_movement: glm::Mat4 = glm::translation(&movement_coordinates);
                let xy_rotation: glm::Mat4 =  glm::rotation(-rotation_coordinates[1], &glm::vec3(1.0, 0.0, 0.0))
                * glm::rotation(rotation_coordinates[0], &glm::vec3(0.0, 1.0, 0.0));
                let view:  glm::Mat4 = xy_rotation * xyz_movement;

                // Projection
                let projection: glm::Mat4 = glm::perspective(SCREEN_H as f32 / SCREEN_W as f32, 0.52, 1.0, 1000.0);  // NB flips the z-axis

                // Putting it all together
                let mut mvp: glm::Mat4 =  projection * view * model;

                gl::ClearColor(0.5, 0.2, 0.7, 1.0);
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
                //gl::Disable(gl::CULL_FACE); // Shows backface of primitves, NB! costs memory
                
                for i in 0..5 {
                    update_node_transformations(&mut *root, &model, elapsed + (i as f32) * 1.5, &shaders);
                    draw_scene(&root, &mut mvp, &shaders);
                }
            }

            context.swap_buffers().unwrap();
        }
    });

    // Keep track of the health of the rendering thread
    let render_thread_healthy = Arc::new(RwLock::new(true));
    let render_thread_watchdog = Arc::clone(&render_thread_healthy);
    thread::spawn(move || {
        if !render_thread.join().is_ok() {
            if let Ok(mut health) = render_thread_watchdog.write() {
                println!("Render thread panicked!");
                *health = false;
            }
        }
    });

    // Start the event loop -- This is where window events get handled
    el.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        // Terminate program if render thread panics
        if let Ok(health) = render_thread_healthy.read() {
            if *health == false {
                *control_flow = ControlFlow::Exit;
            }
        }

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            },
            // Keep track of currently pressed keys to send to the rendering thread
            Event::WindowEvent { event: WindowEvent::KeyboardInput {
                input: KeyboardInput { state: key_state, virtual_keycode: Some(keycode), .. }, .. }, .. } => {

                if let Ok(mut keys) = arc_pressed_keys.lock() {
                    match key_state {
                        Released => {
                            if keys.contains(&keycode) {
                                let i = keys.iter().position(|&k| k == keycode).unwrap();
                                keys.remove(i);
                            }
                        },
                        Pressed => {
                            if !keys.contains(&keycode) {
                                keys.push(keycode);
                            }
                        }
                    }
                }

                // Handle escape separately
                match keycode {
                    Escape => {
                        *control_flow = ControlFlow::Exit;
                    },
                    // Q => {
                    //     *control_flow = ControlFlow::Exit;
                    // }
                    _ => { }
                }
            },
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                // Accumulate mouse movement
                if let Ok(mut position) = arc_mouse_delta.lock() {
                    *position = (position.0 + delta.0 as f32, position.1 + delta.1 as f32);
                }
            },
            _ => { }
        }
    });
}
