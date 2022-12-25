// Vertex shader
struct VetexInput {
	@location(0) position: vec3<f32>,
	@location(1) tex_coords: vec2<f32>,
	@location(2) normal: vec3<f32>,
};

struct VertexOutput {
	@builtin(position) clip_position : vec4<f32>,
	@location(0) tex_coords: vec2<f32>,
	@location(1) world_normal: vec3<f32>,
	@location(2) world_position: vec3<f32>,
};

struct CameraUniform {
	view_pos: vec4<f32>,
	view_proj: mat4x4<f32>,
};

@group(1) @binding(0)
var <uniform> camera: CameraUniform;

struct InstanceInput {
	@location(5) model_matrix_0: vec4<f32>,
	@location(6) model_matrix_1: vec4<f32>,
	@location(7) model_matrix_2: vec4<f32>,
	@location(8) model_matrix_3: vec4<f32>,
	@location(9) normal_matrix_0: vec3<f32>,
	@location(10) normal_matrix_1: vec3<f32>,
	@location(11) normal_matrix_2: vec3<f32>,
}

@vertex
fn vs_main(
	model: VetexInput,
	instance: InstanceInput,
) -> VertexOutput {
	let model_matrix = mat4x4<f32>(
		instance.model_matrix_0,
		instance.model_matrix_1,
		instance.model_matrix_2,
		instance.model_matrix_3,
	);

	let nomal_matrx = mat3x3<f32>(
		instance.normal_matrix_0,
		instance.normal_matrix_1,
		instance.normal_matrix_2,
	);
	var out: VertexOutput;
	out.tex_coords = model.tex_coords;
	out.world_normal = nomal_matrx * model.normal;
	var world_position: vec4<f32> = model_matrix * vec4<f32>(model.position, 1.0f);	
	out.world_position = world_position.xyz;
	out.clip_position = camera.view_proj * world_position;
	return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

struct Light {
	position: vec3<f32>,
	color: vec3<f32>,
}

@group(2) @binding(0)
var<uniform>  light: Light;

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);

	let ambient_strenght = 0.1;
	let ambient_color  = light.color * ambient_strenght;

	let light_dir = normalize(light.position - in.world_position);
	let diffuse_strenght = max(dot(in.world_normal, light_dir), 0.0);
	let diffuse_color = light.color * diffuse_strenght;

	let view_dir = normalize(camera.view_pos.xyz - in.world_position);
	let half_dir = normalize(view_dir + light_dir);
	let reflect_dir = reflect(-light_dir, in.world_normal);

	let specular_strenght = pow(max(dot(in.world_normal, half_dir), 0.0), 32.0);
	let speculer_color = specular_strenght * light.color;

	let result = (ambient_color + diffuse_color + speculer_color) * object_color.xyz;
	//let result = speculer_color;
	//let result = (diffuse_color) * object_color.xyz;

	return vec4<f32>(result, object_color.a);
}