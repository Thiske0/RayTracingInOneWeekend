use clap::Parser;

use grid_nd::GridND;
use rand::rngs::ThreadRng;
use simple_ray_tracer::{
    Result,
    raytracer::{
        camera::Camera,
        options::{Options, RenderOptions},
    },
};

use simple_ray_tracer_kernels::{
    color::Color,
    hitable::HitKind,
    hitable_list_builder::HitableListBuilder,
    materials::{dielectric::Dielectric, lambertian::Lambertian, metal::Metal},
    random::random_single,
    sphere::Sphere,
    textures::{checker::CheckerTexture, image::ImageTexture, solid::SolidTexture},
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
            let sphere_material = Lambertian::new(SolidTexture::new(albedo).into());
            let end = center + Vec3::new(0.0, random_single(0.0..0.2, &mut rng), 0.0);
            Some(Sphere::new_moving(center, end, 0.2, sphere_material).into())
        } else if choose_mat < 0.95 {
            // metal
            let albedo = Color::random(&mut rng) / 2.0 + 0.5;
            let fuzz = random_single(0.0..0.5, &mut rng);
            let sphere_material = Metal::new(SolidTexture::new(albedo).into(), fuzz);
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

#[allow(unused)]
fn bouncing_spheres<'a>() -> Result<HitableListBuilder<'a>> {
    let mut world = HitableListBuilder::new();

    let checker_texture =
        CheckerTexture::new(Color::new(0.2, 0.3, 0.1), Color::new(0.9, 0.9, 0.9), 0.32);
    let ground_material = Lambertian::new(checker_texture.into());
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
    let spheres = spheres.subdivide(&[2]);

    world.add_unrolled(spheres.into());

    let material1a = Dielectric::new(1.5);
    world.add(Sphere::new_static(Point3::new(0.0, 1.0, 0.0), 1.0, material1a).into());

    let material1b = Dielectric::new(1.0 / 1.5);
    world.add(Sphere::new_static(Point3::new(0.0, 1.0, 0.0), 0.8, material1b).into());

    let material2 = Lambertian::new(SolidTexture::new(Color::new(0.4, 0.2, 0.1)).into());
    world.add(Sphere::new_static(Point3::new(-4.0, 1.0, 0.0), 1.0, material2).into());

    let material3 = Metal::new(SolidTexture::new(Color::new(0.7, 0.6, 0.5)).into(), 0.0);
    world.add(Sphere::new_static(Point3::new(4.0, 1.0, 0.0), 1.0, material3).into());
    Ok(world)
}

fn earth<'a>(
    options: &mut RenderOptions,
    earth_image: &'a GridND<Color, 2>,
) -> Result<HitableListBuilder<'a>> {
    let earth_texture = ImageTexture::new(&earth_image);
    let earth_surface = Lambertian::new(earth_texture.into());
    let globe = Sphere::new_static(Point3::new(0.0, 0.0, 0.0), 2.0, earth_surface);

    options.vertical_fov = 20.0;
    options.lookfrom = Point3::new(12.0, 0.0, 0.0);
    options.lookat = Point3::new(0.0, 0.0, 0.0);
    options.vup = Vec3::new(0.0, 1.0, 0.0);

    options.defocus_angle = 0.0;

    let mut world = HitableListBuilder::new();
    world.add(globe.into());
    Ok(world)
}

fn main() -> Result<()> {
    // Parse command line options
    let mut options = Options::parse();

    // World setup
    //let mut world = bouncing_spheres()?;
    let earth_image = ImageTexture::from_file("data/earthmap.jpg")?;
    let mut world = earth(&mut options.render, &earth_image)?;

    // Camera setup
    let camera = Camera::new(options.render);

    // Time duration
    let start = std::time::Instant::now();

    camera.render((&mut world).into())?;

    let duration = start.elapsed();
    println!("Render time: {:?}", duration);

    Ok(())
}
