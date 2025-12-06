use clap::Parser;
use core::panic;

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
        constant_medium::ConstantMedium,
        hitable_list_builder::{HitableListBuilder, SlitMethod},
        object_parser::parse_obj,
        planar::{Quad, Triangle, make_box},
        rotate::Rotate,
        scale::Scale,
        sphere::Sphere,
        translate::Translate,
    },
    materials::{
        dielectric::Dielectric, diffuse_light::DiffuseLight, isotropic::Isotropic,
        lambertian::Lambertian, metal::Metal,
    },
    random::RandomRange,
    textures::{
        checker::CheckerTexture, image::ImageTexture, perlin::PerlinTexture, solid::SolidTexture,
    },
    vec3::{Axis, Point3, Real, Vec3},
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

fn bouncing_spheres<'a>(options: &mut RenderOptions) -> HitableListBuilder<'a> {
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
    let spheres = spheres.subdivide(&[2], &SlitMethod::Middle, false);

    world.add_unrolled(spheres.into());

    let material1a = Dielectric::new(1.5);
    world.add(Sphere::new_static(Point3::new(0.0, 1.0, 0.0), 1.0, material1a).into());

    let material1b = Dielectric::new(1.0 / 1.5);
    world.add(Sphere::new_static(Point3::new(0.0, 1.0, 0.0), 0.8, material1b).into());

    let material2 = Lambertian::new(SolidTexture::new(Color::new(0.4, 0.2, 0.1)).into());
    world.add(Sphere::new_static(Point3::new(-4.0, 1.0, 0.0), 1.0, material2).into());

    let material3 = Metal::new(SolidTexture::new(Color::new(0.7, 0.6, 0.5)).into(), 0.0);
    world.add(Sphere::new_static(Point3::new(4.0, 1.0, 0.0), 1.0, material3).into());

    options.background = Color::new(0.7, 0.8, 1.0);

    world
}

fn earth<'a>(
    options: &mut RenderOptions,
    earth_image: &'a GridND<Color, 2>,
) -> HitableListBuilder<'a> {
    let earth_texture = ImageTexture::new(&earth_image);
    let earth_surface = Lambertian::new(earth_texture);
    let globe = Sphere::new_static(Point3::new(0.0, 0.0, 0.0), 2.0, earth_surface);

    options.vertical_fov = 20.0;
    options.lookfrom = Point3::new(12.0, 0.0, 0.0);
    options.lookat = Point3::new(0.0, 0.0, 0.0);
    options.vup = Vec3::new(0.0, 1.0, 0.0);

    options.defocus_angle = 0.0;
    options.background = Color::new(0.7, 0.8, 1.0);

    let mut world: HitableListBuilder<'_> = HitableListBuilder::new();
    world.add(globe);
    world
}

fn perlin_spheres<'a>(options: &mut RenderOptions) -> HitableListBuilder<'a> {
    let mut world: HitableListBuilder<'_> = HitableListBuilder::new();

    world.add(Sphere::new_static(
        Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        Lambertian::new(PerlinTexture::new(4.0).into()),
    ));
    world.add(Sphere::new_static(
        Point3::new(0.0, 2.0, 0.0),
        2.0,
        Lambertian::new(PerlinTexture::new(8.0).into()),
    ));

    options.vertical_fov = 20.0;
    options.lookfrom = Point3::new(13.0, 2.0, 3.0);
    options.lookat = Point3::new(0.0, 0.0, 0.0);
    options.vup = Vec3::new(0.0, 1.0, 0.0);

    options.defocus_angle = 0.0;
    options.background = Color::new(0.7, 0.8, 1.0);

    world
}

fn planar<'a>(options: &mut RenderOptions) -> HitableListBuilder<'a> {
    let mut world: HitableListBuilder<'_> = HitableListBuilder::new();

    // Materials
    let red = Lambertian::new(SolidTexture::new(Color::new(1.0, 0.2, 0.2)).into());
    let green = Lambertian::new(SolidTexture::new(Color::new(0.2, 1.0, 0.2)).into());
    let blue = Lambertian::new(SolidTexture::new(Color::new(0.2, 0.2, 1.0)).into());
    let orange = Lambertian::new(SolidTexture::new(Color::new(1.0, 0.5, 0.0)).into());
    let teal = Lambertian::new(SolidTexture::new(Color::new(0.2, 0.8, 0.8)).into());

    // Quads
    world.add(Quad::new(
        Point3::new(-3.0, -2.0, 5.0),
        Vec3::new(0.0, 0.0, -4.0),
        Vec3::new(0.0, 4.0, 0.0),
        red,
    ));
    world.add(Quad::new(
        Point3::new(-2.0, -2.0, 0.0),
        Vec3::new(4.0, 0.0, 0.0),
        Vec3::new(0.0, 4.0, 0.0),
        green,
    ));
    world.add(Quad::new(
        Point3::new(3.0, -2.0, 1.0),
        Vec3::new(0.0, 0.0, 4.0),
        Vec3::new(0.0, 4.0, 0.0),
        blue,
    ));
    world.add(Triangle::new(
        Point3::new(-2.0, 3.0, 5.0),
        Point3::new(2.0, 3.0, 5.0),
        Point3::new(2.0, 3.0, 1.0),
        teal.clone(),
    ));
    world.add(Triangle::new(
        Point3::new(-2.0, 3.0, 5.0),
        Point3::new(2.0, 3.0, 1.0),
        Point3::new(-2.0, 3.0, 1.0),
        orange.clone(),
    ));
    world.add(Triangle::new(
        Point3::new(-2.0, -3.0, 5.0),
        Point3::new(2.0, -3.0, 5.0),
        Point3::new(-2.0, -3.0, 1.0),
        teal,
    ));
    world.add(Triangle::new(
        Point3::new(2.0, -3.0, 5.0),
        Point3::new(2.0, -3.0, 1.0),
        Point3::new(-2.0, -3.0, 1.0),
        orange,
    ));

    options.vertical_fov = 80.0;
    options.lookfrom = Point3::new(0.0, 0.0, 9.0);
    options.lookat = Point3::new(0.0, 0.0, 0.0);
    options.vup = Vec3::new(0.0, 1.0, 0.0);

    options.defocus_angle = 0.0;
    options.background = Color::new(0.7, 0.8, 1.0);

    world
}

fn lights<'a>(options: &mut RenderOptions) -> HitableListBuilder<'a> {
    let mut world = perlin_spheres(options);

    let difflight = DiffuseLight::new(SolidTexture::new(Color::new(4.0, 4.0, 4.0)));
    world.add(Quad::new(
        Point3::new(3.0, 1.0, -2.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        difflight,
    ));

    options.lookfrom = Point3::new(26.0, 3.0, 6.0);
    options.background = Color::black();

    world
}

fn make_cornell_box<'a>(world: &mut HitableListBuilder<'a>, options: &mut RenderOptions) {
    let red = Lambertian::new(SolidTexture::new(Color::new(0.65, 0.05, 0.05)));
    let green = Lambertian::new(SolidTexture::new(Color::new(0.12, 0.45, 0.15)));
    let light = DiffuseLight::new(SolidTexture::new(Color::new(15.0, 15.0, 15.0)));
    let white = Lambertian::new(SolidTexture::new(Color::new(0.73, 0.73, 0.73)));

    world.add(Quad::new(
        Point3::new(555.0, 0.0, 0.0),
        Vec3::new(0.0, 555.0, 0.0),
        Vec3::new(0.0, 0.0, 555.0),
        green,
    ));
    world.add(Quad::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 555.0, 0.0),
        Vec3::new(0.0, 0.0, 555.0),
        red,
    ));
    world.add(Quad::new(
        Point3::new(343.0, 554.0, 332.0),
        Vec3::new(-130.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -105.0),
        light,
    ));
    world.add(Quad::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(555.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 555.0),
        white.clone(),
    ));
    world.add(Quad::new(
        Point3::new(555.0, 555.0, 555.0),
        Vec3::new(-555.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -555.0),
        white.clone(),
    ));
    world.add(Quad::new(
        Point3::new(0.0, 0.0, 555.0),
        Vec3::new(555.0, 0.0, 0.0),
        Vec3::new(0.0, 555.0, 0.0),
        white.clone(),
    ));

    options.vertical_fov = 40.0;
    options.lookfrom = Point3::new(278.0, 278.0, -800.0);
    options.lookat = Point3::new(278.0, 278.0, 0.0);
    options.vup = Vec3::new(0.0, 1.0, 0.0);
    options.width = options.height;

    options.defocus_angle = 0.0;
}

fn cornell_box<'a>(options: &mut RenderOptions) -> HitableListBuilder<'a> {
    let mut world = HitableListBuilder::new();

    let white = Lambertian::new(SolidTexture::new(Color::new(0.73, 0.73, 0.73)));

    world.add(Translate::new_owned(
        Vec3::new(265.0, 0.0, 295.0),
        Rotate::new_owned(
            Axis::Y,
            15.0_f32.to_radians(),
            make_box(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(165.0, 330.0, 165.0),
                white.clone(),
            )
            .into(),
        ),
    ));
    world.add(Translate::new_owned(
        Vec3::new(130.0, 0.0, 65.0),
        Rotate::new_owned(
            Axis::Y,
            -18.0_f32.to_radians(),
            make_box(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(165.0, 165.0, 165.0),
                white.clone(),
            )
            .into(),
        ),
    ));

    make_cornell_box(&mut world, options);
    options.samples_per_pixel = 200;

    world
}

fn cornell_box_smoke<'a>(options: &mut RenderOptions) -> HitableListBuilder<'a> {
    let mut world = HitableListBuilder::new();

    let red = Lambertian::new(SolidTexture::new(Color::new(0.65, 0.05, 0.05)));
    let green = Lambertian::new(SolidTexture::new(Color::new(0.12, 0.45, 0.15)));
    let light = DiffuseLight::new(SolidTexture::new(Color::new(7.0, 7.0, 7.0)));
    let white = Lambertian::new(SolidTexture::new(Color::new(0.73, 0.73, 0.73)));
    let smoke = Dielectric::new(1.5);

    world.add(Quad::new(
        Point3::new(555.0, 0.0, 0.0),
        Vec3::new(0.0, 555.0, 0.0),
        Vec3::new(0.0, 0.0, 555.0),
        green,
    ));
    world.add(Quad::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 555.0, 0.0),
        Vec3::new(0.0, 0.0, 555.0),
        red,
    ));
    world.add(Quad::new(
        Point3::new(343.0, 554.0, 332.0),
        Vec3::new(-130.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -105.0),
        light,
    ));
    world.add(Quad::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(555.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 555.0),
        white.clone(),
    ));
    world.add(Quad::new(
        Point3::new(555.0, 555.0, 555.0),
        Vec3::new(-555.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -555.0),
        white.clone(),
    ));
    world.add(Quad::new(
        Point3::new(0.0, 0.0, 555.0),
        Vec3::new(555.0, 0.0, 0.0),
        Vec3::new(0.0, 555.0, 0.0),
        white.clone(),
    ));

    let smoke_box_1 = Translate::new_owned(
        Vec3::new(265.0, 0.0, 295.0),
        Rotate::new_owned(
            Axis::Y,
            15.0_f32.to_radians(),
            make_box(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(165.0, 330.0, 165.0),
                smoke.clone(),
            )
            .into(),
        ),
    );

    world.add(ConstantMedium::new_owned(
        0.01,
        smoke_box_1,
        Isotropic::new(Color::black()),
    ));

    let smoke_box_2 = Translate::new_owned(
        Vec3::new(130.0, 1.0, 65.0),
        Rotate::new_owned(
            Axis::Y,
            -18.0_f32.to_radians(),
            make_box(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(165.0, 165.0, 165.0),
                smoke.clone(),
            )
            .into(),
        ),
    );

    world.add(ConstantMedium::new_owned(
        0.01,
        smoke_box_2,
        Isotropic::new(Color::white()),
    ));

    options.vertical_fov = 40.0;
    options.lookfrom = Point3::new(278.0, 278.0, -800.0);
    options.lookat = Point3::new(278.0, 278.0, 0.0);
    options.vup = Vec3::new(0.0, 1.0, 0.0);
    options.width = options.height;
    options.samples_per_pixel = 200;

    options.defocus_angle = 0.0;

    world
}

fn final_scene<'a>(
    options: &mut RenderOptions,
    earth_image: &'a GridND<Color, 2>,
) -> HitableListBuilder<'a> {
    let mut world = HitableListBuilder::new();
    let mut floor = HitableListBuilder::new();
    let ground = Lambertian::new(SolidTexture::new(Color::new(0.48, 0.83, 0.53)));

    let mut rng = rand::rng();
    let boxes_per_side = 20;
    for i in 0..boxes_per_side {
        for j in 0..boxes_per_side {
            let size = 100.0;
            let x0 = -1000.0 + i as f32 * size;
            let z0 = -1000.0 + j as f32 * size;
            let y1 = rng.random_range(1.0..101.0);
            floor.add_unrolled(make_box(
                Point3::new(x0, 0.0, z0),
                Point3::new(x0 + size, y1, z0 + size),
                ground.clone(),
            ));
        }
    }
    let subdivided_floor = floor.subdivide(&[2, 4], &SlitMethod::Middle, false).into();
    world.add(subdivided_floor);

    world.add(Quad::new(
        Point3::new(123.0, 554.0, 147.0),
        Vec3::new(300.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 265.0),
        DiffuseLight::new(SolidTexture::new(Color::new(7.0, 7.0, 7.0))),
    ));

    let start = Point3::new(400.0, 400.0, 200.0);
    let end = start + Vec3::new(30.0, 0.0, 0.0);
    world.add(
        Sphere::new_moving(
            start,
            end,
            50.0,
            Lambertian::new(SolidTexture::new(Color::new(0.7, 0.3, 0.1))),
        )
        .into(),
    );

    world.add(Sphere::new_static(
        Point3::new(260.0, 150.0, 45.0),
        50.0,
        Dielectric::new(1.5),
    ));
    world.add(Sphere::new_static(
        Point3::new(0.0, 150.0, 145.0),
        50.0,
        Metal::new(SolidTexture::new(Color::new(0.8, 0.8, 0.9)), 1.0),
    ));

    //Dielectric with smoke inside
    world.add(Sphere::new_static(
        Point3::new(360.0, 150.0, 145.0),
        70.0,
        Dielectric::new(1.5),
    ));
    let boundary = Sphere::new_static(
        Point3::new(360.0, 150.0, 145.0),
        70.0 - 1e-3,
        Dielectric::new(1.5),
    );
    world.add(ConstantMedium::new_owned(
        0.2,
        boundary,
        Isotropic::new(Color::new(0.2, 0.4, 0.9)),
    ));

    let boundary = Sphere::new_static(Point3::new(0.0, 0.0, 0.0), 5000.0, Dielectric::new(1.5));
    world.add(ConstantMedium::new_owned(
        0.0001,
        boundary,
        Isotropic::new(Color::new(1.0, 1.0, 1.0)),
    ));

    world.add(Sphere::new_static(
        Point3::new(400.0, 200.0, 400.0),
        100.0,
        Lambertian::new(ImageTexture::new(&earth_image)),
    ));
    world.add(Sphere::new_static(
        Point3::new(220.0, 280.0, 300.0),
        80.0,
        Lambertian::new(PerlinTexture::new(0.2)),
    ));

    let mut spheres = HitableListBuilder::new();
    let white = Lambertian::new(SolidTexture::new(Color::new(0.73, 0.73, 0.73)));
    let ns = 1000;
    for _ in 0..ns {
        spheres.add(Sphere::new_static(
            Point3::random(0.0..165.0, &mut rng),
            10.0,
            white.clone(),
        ));
    }
    let subdivided_spheres = spheres
        .subdivide(&[3, 3], &SlitMethod::Middle, false)
        .into();

    world.add(Translate::new_owned(
        Vec3::new(-100.0, 270.0, 395.0),
        Rotate::new_owned(Axis::Y, 15.0, subdivided_spheres),
    ));

    options.vertical_fov = 40.0;
    options.lookfrom = Point3::new(478.0, 278.0, -600.0);
    options.lookat = Point3::new(278.0, 278.0, 0.0);
    options.vup = Vec3::new(0.0, 1.0, 0.0);
    options.width = options.height;
    options.samples_per_pixel = 50;

    options.defocus_angle = 0.0;

    world
}

fn bunny<'a>(options: &mut RenderOptions) -> HitableListBuilder<'a> {
    let mut world = HitableListBuilder::new();

    // 4_968 faces
    let bunny = parse_obj(
        "data/bunny.obj",
        Lambertian::new(SolidTexture::new(Color::new(0.0, 0.0, 0.8)).into()),
    )
    .expect("Failed to parse bunny.obj");

    let bunny = bunny.subdivide_by4(5, SlitMethod::Sah, false);

    world.add(Translate::new_owned(
        Vec3::new(277.5, -50.0, 277.5),
        Rotate::new_owned(
            Axis::Y,
            180.0_f32.to_radians(),
            Scale::new_owned_same(1500.0, bunny.into()),
        ),
    ));

    make_cornell_box(&mut world, options);

    options.samples_per_pixel = 50;

    world
}

fn dragon<'a>(options: &mut RenderOptions) -> HitableListBuilder<'a> {
    let mut world = HitableListBuilder::new();

    // 87_130 faces
    let dragon = parse_obj(
        "data/Dragon_80K.obj",
        Lambertian::new(SolidTexture::new(Color::new(0.0, 0.0, 0.8)).into()),
    )
    .expect("Failed to parse Dragon_80K.obj");

    let dragon = dragon.subdivide_by4(8, SlitMethod::Sah, false);

    world.add(Translate::new_owned(
        Vec3::new(277.5, 100.0, 277.5),
        Rotate::new_owned(
            Axis::Y,
            -90.0_f32.to_radians(),
            Scale::new_owned_same(400.0, dragon.into()),
        ),
    ));

    make_cornell_box(&mut world, options);

    options.samples_per_pixel = 50;

    world
}

fn perf_test<'a>(options: &mut RenderOptions) -> HitableListBuilder<'a> {
    let mut world = HitableListBuilder::new();

    // 87_130 faces
    let dragon = parse_obj(
        "data/Dragon_80K.obj",
        Lambertian::new(SolidTexture::new(Color::new(0.0, 0.0, 0.8)).into()),
    )
    .expect("Failed to parse Dragon_80K.obj");

    // Time duration
    let start = std::time::Instant::now();

    // Make BVH
    let dragon = dragon.subdivide_by4(8, SlitMethod::SahBy4, true);

    let duration = start.elapsed();
    println!("BVH build time: {:?}", duration);

    world.add(Translate::new_owned(
        Vec3::new(277.5, 100.0, 277.5),
        Rotate::new_owned(
            Axis::Y,
            -90.0_f32.to_radians(),
            Scale::new_owned_same(400.0, dragon.into()),
        ),
    ));

    make_cornell_box(&mut world, options);

    options.width = 2160;
    options.height = 2160;
    options.samples_per_pixel = 200;
    options.max_depth = 50;
    options.calculate_noise = false;

    world
}

fn main() -> Result<()> {
    // Parse command line options
    let mut options = Options::parse();

    let earth_image = ImageTexture::from_file("data/earthmap.jpg")?;

    let scene = 11;
    let world = match scene {
        1 => bouncing_spheres(&mut options.render),
        2 => earth(&mut options.render, &earth_image),
        3 => perlin_spheres(&mut options.render),
        4 => planar(&mut options.render),
        5 => lights(&mut options.render),
        6 => cornell_box(&mut options.render),
        7 => cornell_box_smoke(&mut options.render),
        8 => final_scene(&mut options.render, &earth_image),
        9 => bunny(&mut options.render),
        10 => dragon(&mut options.render),
        11 => perf_test(&mut options.render),
        _ => {
            panic!("Unknown scene {}", scene);
        }
    };

    let has_noise_calculation = options.render.calculate_noise;

    // Camera setup
    let camera = Camera::new(options.render);

    // Time duration
    let start = std::time::Instant::now();

    camera.render(world.into())?;

    let mut duration = start.elapsed();
    if has_noise_calculation {
        duration /= 2; // Rough estimate since noise calculation does two renders
    }
    println!("Render time: {:?}", duration);

    Ok(())
}
