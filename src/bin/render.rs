use clap::Parser;

use simple_ray_tracer::{
    Result,
    raytracer::{
        camera::Camera,
        color::Color,
        hitable_list::HitableList,
        materials::{lambertian::Lambertian, metal::Metal},
        options::Options,
        sphere::Sphere,
        vec3::Point3,
    },
};

fn main() -> Result<()> {
    // Parse command line options
    let options = Options::parse();

    // Materials
    let material_ground = Lambertian::new(Color::new(0.8, 0.8, 0.0));
    let material_center = Lambertian::new(Color::new(0.1, 0.2, 0.5));
    let material_left = Metal::new(Color::new(0.8, 0.8, 0.8), 0.2);
    let material_right = Metal::new(Color::new(0.8, 0.6, 0.2), 0.5);

    // World
    let mut world = HitableList::new();
    world.add(Sphere::new(
        Point3::new(0.0, -100.5, -1.0),
        100.0,
        material_ground,
    ));
    world.add(Sphere::new(
        Point3::new(0.0, 0.0, -1.2),
        0.5,
        material_center,
    ));
    world.add(Sphere::new(
        Point3::new(-1.0, 0.0, -1.0),
        0.5,
        material_left,
    ));
    world.add(Sphere::new(
        Point3::new(1.0, 0.0, -1.0),
        0.5,
        material_right,
    ));

    // Camera setup
    let camera = Camera::new(Point3::new(0.0, 0.0, 0.0), options.render);

    camera.render(&world)?;
    Ok(())
}
