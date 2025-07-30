use clap::Parser;

use simple_ray_tracer::{
    Result,
    raytracer::{
        camera::Camera, hitable_list::HitableList, options::Options, sphere::Sphere, vec3::Point3,
    },
};

fn main() -> Result<()> {
    // Parse command line options
    let options = Options::parse();

    // World
    let mut world = HitableList::new();
    world.add(Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5));
    world.add(Sphere::new(Point3::new(0.0, -100.5, -1.0), 100.0));

    // Camera setup
    let camera = Camera::new(Point3::new(0.0, 0.0, 0.0), options.render);

    camera.render(&world)?;
    Ok(())
}
