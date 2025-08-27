use core::panic;

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
    hitables::{
        HitKind,
        hitable_list_builder::HitableListBuilder,
        planar::{Quad, Triangle},
        sphere::Sphere,
    },
    materials::{dielectric::Dielectric, lambertian::Lambertian, metal::Metal},
    random::RandomRange,
    textures::{
        checker::CheckerTexture, image::ImageTexture, perlin::PerlinTexture, solid::SolidTexture,
    },
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
            let end = center + Vec3::new(0.0, rng.random_range(0.0..0.2), 0.0);
            Some(Sphere::new_moving(center, end, 0.2, sphere_material).into())
        } else if choose_mat < 0.95 {
            // metal
            let albedo = Color::random(&mut rng) / 2.0 + 0.5;
            let fuzz = rng.random_range(0.0..0.5);
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
) -> HitableListBuilder<'a> {
    let earth_texture = ImageTexture::new(&earth_image);
    let earth_surface = Lambertian::new(earth_texture.into());
    let globe = Sphere::new_static(Point3::new(0.0, 0.0, 0.0), 2.0, earth_surface);

    options.vertical_fov = 20.0;
    options.lookfrom = Point3::new(12.0, 0.0, 0.0);
    options.lookat = Point3::new(0.0, 0.0, 0.0);
    options.vup = Vec3::new(0.0, 1.0, 0.0);

    options.defocus_angle = 0.0;

    let mut world: HitableListBuilder<'_> = HitableListBuilder::new();
    world.add(globe.into());
    world
}

fn perlin_spheres<'a>(options: &mut RenderOptions) -> HitableListBuilder<'a> {
    let mut world: HitableListBuilder<'_> = HitableListBuilder::new();

    world.add(
        Sphere::new_static(
            Point3::new(0.0, -1000.0, 0.0),
            1000.0,
            Lambertian::new(PerlinTexture::new(4.0).into()),
        )
        .into(),
    );
    world.add(
        Sphere::new_static(
            Point3::new(0.0, 2.0, 0.0),
            2.0,
            Lambertian::new(PerlinTexture::new(8.0).into()),
        )
        .into(),
    );

    options.vertical_fov = 20.0;
    options.lookfrom = Point3::new(13.0, 2.0, 3.0);
    options.lookat = Point3::new(0.0, 0.0, 0.0);
    options.vup = Vec3::new(0.0, 1.0, 0.0);

    options.defocus_angle = 0.0;

    world
}

fn planar<'a>(options: &mut RenderOptions) -> HitableListBuilder<'a> {
    let mut world: HitableListBuilder<'_> = HitableListBuilder::new();

    // Materials
    let left_red = Lambertian::new(SolidTexture::new(Color::new(1.0, 0.2, 0.2)).into());
    let back_green = Lambertian::new(SolidTexture::new(Color::new(0.2, 1.0, 0.2)).into());
    let right_blue = Lambertian::new(SolidTexture::new(Color::new(0.2, 0.2, 1.0)).into());
    let upper_orange = Lambertian::new(SolidTexture::new(Color::new(1.0, 0.5, 0.0)).into());
    let upper_teal = Lambertian::new(SolidTexture::new(Color::new(0.2, 0.8, 0.8)).into());
    let lower_orange = Lambertian::new(SolidTexture::new(Color::new(1.0, 0.5, 0.0)).into());
    let lower_teal = Lambertian::new(SolidTexture::new(Color::new(0.2, 0.8, 0.8)).into());

    // Quads
    world.add(
        Quad::new(
            Point3::new(-3.0, -2.0, 5.0),
            Vec3::new(0.0, 0.0, -4.0),
            Vec3::new(0.0, 4.0, 0.0),
            left_red,
        )
        .into(),
    );
    world.add(
        Quad::new(
            Point3::new(-2.0, -2.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            back_green,
        )
        .into(),
    );
    world.add(
        Quad::new(
            Point3::new(3.0, -2.0, 1.0),
            Vec3::new(0.0, 0.0, 4.0),
            Vec3::new(0.0, 4.0, 0.0),
            right_blue,
        )
        .into(),
    );
    world.add(
        Triangle::new(
            Point3::new(-2.0, 3.0, 5.0),
            Point3::new(2.0, 3.0, 5.0),
            Point3::new(2.0, 3.0, 1.0),
            upper_teal,
        )
        .into(),
    );
    world.add(
        Triangle::new(
            Point3::new(-2.0, 3.0, 5.0),
            Point3::new(2.0, 3.0, 1.0),
            Point3::new(-2.0, 3.0, 1.0),
            upper_orange,
        )
        .into(),
    );
    world.add(
        Triangle::new(
            Point3::new(-2.0, -3.0, 5.0),
            Point3::new(2.0, -3.0, 5.0),
            Point3::new(-2.0, -3.0, 1.0),
            lower_teal,
        )
        .into(),
    );
    world.add(
        Triangle::new(
            Point3::new(2.0, -3.0, 5.0),
            Point3::new(2.0, -3.0, 1.0),
            Point3::new(-2.0, -3.0, 1.0),
            lower_orange,
        )
        .into(),
    );

    options.vertical_fov = 80.0;
    options.lookfrom = Point3::new(0.0, 0.0, 9.0);
    options.lookat = Point3::new(0.0, 0.0, 0.0);
    options.vup = Vec3::new(0.0, 1.0, 0.0);

    options.defocus_angle = 0.0;

    world
}

fn main() -> Result<()> {
    // Parse command line options
    let mut options = Options::parse();

    let earth_image = ImageTexture::from_file("data/earthmap.jpg")?;

    let scene = 4;
    let mut world = match scene {
        1 => bouncing_spheres()?,
        2 => earth(&mut options.render, &earth_image),
        3 => perlin_spheres(&mut options.render),
        4 => planar(&mut options.render),
        _ => {
            panic!("Unknown scene {}", scene);
        }
    };

    // Camera setup
    let camera = Camera::new(options.render);

    // Time duration
    let start = std::time::Instant::now();

    camera.render((&mut world).into())?;

    let duration = start.elapsed();
    println!("Render time: {:?}", duration);

    Ok(())
}
