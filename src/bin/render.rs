use clap::Parser;

use rand::rngs::ThreadRng;
use simple_ray_tracer::{
    Result,
    raytracer::{camera::Camera, options::Options},
};

use simple_ray_tracer_kernels::{
    color::Color,
    hitable::HitKind,
    hitable_list::HitableListBuilder,
    materials::{dielectric::Dielectric, lambertian::Lambertian, metal::Metal},
    random::random_single,
    sphere::Sphere,
    vec3::{Point3, Real, Vec3},
};

fn generate_random_sphere<'a>(x: Real, y: Real, mut rng: ThreadRng) -> Option<HitKind<'a>> {
    let random_vec = Vec3::random(0.0..1.0, &mut rng);
    let choose_mat = random_vec.x;
    let center = Point3::new(x + 0.9 * random_vec.y, 0.2, y + 0.9 * random_vec.z);

    if (center - Point3::new(4.0, 0.2, 0.0)).length() > 0.9 {
        if choose_mat < 0.8 {
            // diffuse
            let albedo = Color::random(&mut rng) * Color::random(&mut rng);
            let sphere_material = Lambertian::new(albedo);
            let end = center + Vec3::new(0.0, random_single(0.0..0.2, &mut rng), 0.0);
            Some(Sphere::new_moving(center, end, 0.2, sphere_material).into())
        } else if choose_mat < 0.95 {
            // metal
            let albedo = Color::random(&mut rng) / 2.0 + 0.5;
            let fuzz = random_single(0.0..0.5, &mut rng);
            let sphere_material = Metal::new(albedo, fuzz);
            Some(Sphere::new_static(center, 0.2, sphere_material).into())
        } else {
            // glass
            let sphere_material = Dielectric::new(1.5);
            Some(Sphere::new_static(center, 0.2, sphere_material).into())
        }
    } else {
        None
    }
}

fn main() -> Result<()> {
    // Parse command line options
    let options = Options::parse();

    // World
    let mut world = HitableListBuilder::new();

    let ground_material = Lambertian::new(Color::new(0.5, 0.5, 0.5));
    world.add(Sphere::new_static(Point3::new(0.0, -1000.0, 0.0), 1000.0, ground_material).into());

    let rng = rand::rng();

    let mut spheres = HitableListBuilder::new();

    for a in -11..11 {
        for b in -11..11 {
            if let Some(sphere) = generate_random_sphere(a as Real, b as Real, rng.clone()) {
                spheres.add(sphere);
            }
        }
    }
    world.add_builder(spheres.into());

    let material1a = Dielectric::new(1.5);
    world.add(Sphere::new_static(Point3::new(0.0, 1.0, 0.0), 1.0, material1a).into());

    let material1b = Dielectric::new(1.0 / 1.5);
    world.add(Sphere::new_static(Point3::new(0.0, 1.0, 0.0), 0.8, material1b).into());

    let material2 = Lambertian::new(Color::new(0.4, 0.2, 0.1));
    world.add(Sphere::new_static(Point3::new(-4.0, 1.0, 0.0), 1.0, material2).into());

    let material3 = Metal::new(Color::new(0.7, 0.6, 0.5), 0.0);
    world.add(Sphere::new_static(Point3::new(4.0, 1.0, 0.0), 1.0, material3).into());

    // Camera setup
    let camera = Camera::new(options.render);

    // Time duration
    let start = std::time::Instant::now();

    camera.render(&mut world)?;

    let duration = start.elapsed();
    println!("Render time: {:?}", duration);

    Ok(())
}
