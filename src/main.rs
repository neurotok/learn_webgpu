use learn_webgpu::run;
use pollster;

fn main() {
    pollster::block_on(run());
}
