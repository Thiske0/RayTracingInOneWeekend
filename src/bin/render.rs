use clap::Parser;

use simple_ray_tracer::{
    Result,
    raytracer::{
        camera::Camera,
        color::Color,
        hitable_list::HitableList,
        materials::{dielectric::Dielectric, lambertian::Lambertian, metal::Metal},
        options::Options,
        sphere::Sphere,
        vec3::{Point3, Real, Vec3},
    },
};

fn main() -> Result<()> {
    // Parse command line options
    let options = Options::parse();

    // World
    let mut world = HitableList::new();

    let ground_material = Lambertian::new(Color::new(0.5, 0.5, 0.5));
    world.add(Sphere::new(
        Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        ground_material,
    ));

    for a in -11..11 {
        for b in -11..11 {
            let random_vec = Vec3::random(0.0..1.0);
            let choose_mat = random_vec.x;
            let center = Point3::new(
                a as Real + 0.9 * random_vec.y,
                0.2,
                b as Real + 0.9 * random_vec.z,
            );

            if (center - Point3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                if choose_mat < 0.8 {
                    // diffuse
                    let albedo = Color::random() * Color::random();
                    let sphere_material = Lambertian::new(albedo);
                    world.add(Sphere::new(center, 0.2, sphere_material));
                } else if choose_mat < 0.95 {
                    // metal
                    let albedo = Color::random() / 2.0 + 0.5;
                    let fuzz = Vec3::random(0.0..0.5).x;
                    let sphere_material = Metal::new(albedo, fuzz);
                    world.add(Sphere::new(center, 0.2, sphere_material));
                } else {
                    // glass
                    let sphere_material = Dielectric::new(1.5);
                    world.add(Sphere::new(center, 0.2, sphere_material));
                }
            }
        }
    }

    let material1a = Dielectric::new(1.5);
    world.add(Sphere::new(Point3::new(0.0, 1.0, 0.0), 1.0, material1a));

    let material1b = Dielectric::new(1.0 / 1.5);
    world.add(Sphere::new(Point3::new(0.0, 1.0, 0.0), 0.8, material1b));

    let material2 = Lambertian::new(Color::new(0.4, 0.2, 0.1));
    world.add(Sphere::new(Point3::new(-4.0, 1.0, 0.0), 1.0, material2));

    let material3 = Metal::new(Color::new(0.7, 0.6, 0.5), 0.0);
    world.add(Sphere::new(Point3::new(4.0, 1.0, 0.0), 1.0, material3));

    // Camera setup
    let camera = Camera::new(options.render);

    // Time duration
    let start = std::time::Instant::now();

    camera.render(&world.into())?;

    let duration = start.elapsed();
    println!("Render time: {:?}", duration);

    Ok(())
}
