use crate::random::RandomRange;
use crate::vec3::Vec3;
use crate::{
    color::Color,
    random::Random,
    textures::Texture,
    vec3::{Point3, Real},
};
use gpu_builder::{Builder, DeviceCopyBuilder};
use ref_builder::{RefBuilder, RefBuilderDevice};

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;
#[cfg(not(target_os = "cuda"))]
use std::sync::LazyLock;

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;

type TextureNoise = PerlinNoise<8>;

#[repr(C)]
#[derive(Builder)]
#[use_lifetime("'a")]
pub struct PerlinTexture<'a> {
    scale: Real,
    noise: RefBuilder<'a, TextureNoise>,
}

#[cfg(not(target_os = "cuda"))]
static NOISE: LazyLock<TextureNoise> = LazyLock::new(|| TextureNoise::new(&mut rand::rng()));

#[cfg(not(target_os = "cuda"))]
impl PerlinTexture<'_> {
    pub fn new(scale: Real) -> Self {
        PerlinTexture {
            scale,
            noise: RefBuilder::new(LazyLock::force(&NOISE)),
        }
    }
}

impl<'b> Texture for PerlinTexture<'b> {
    fn color(&self, _u: Real, _v: Real, p: &Point3, _rng: &mut Random) -> Color {
        Color::white() * 0.5 * (1.0 + (self.scale * p.z + 10.0 * self.noise.turb(p, 7)).sin())
    }
}

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy))]
#[derive(DeviceCopyBuilder)]
struct PerlinNoise<const POINT_AMOUNT: usize>
where
    [(); 1 << POINT_AMOUNT]:,
{
    random_vec: [Vec3; 1 << POINT_AMOUNT],
    perm_x: [usize; 1 << POINT_AMOUNT],
    perm_y: [usize; 1 << POINT_AMOUNT],
    perm_z: [usize; 1 << POINT_AMOUNT],
}

impl<const POINT_AMOUNT: usize> PerlinNoise<POINT_AMOUNT>
where
    [(); 1 << POINT_AMOUNT]:,
{
    #[cfg(not(target_os = "cuda"))]
    pub fn new(rng: &mut Random) -> Self {
        let mut result = PerlinNoise {
            random_vec: [Vec3::zero(); 1 << POINT_AMOUNT],
            perm_x: [0; 1 << POINT_AMOUNT],
            perm_y: [0; 1 << POINT_AMOUNT],
            perm_z: [0; 1 << POINT_AMOUNT],
        };

        for value in result.random_vec.iter_mut() {
            *value = Vec3::random_unit(rng);
        }

        Self::perlin_generate_perm(&mut result.perm_x, rng);
        Self::perlin_generate_perm(&mut result.perm_y, rng);
        Self::perlin_generate_perm(&mut result.perm_z, rng);

        result
    }

    pub fn noise(&self, p: &Point3) -> Real {
        let u = p.x - p.x.floor();
        let v = p.y - p.y.floor();
        let w = p.z - p.z.floor();

        let i = (p.x.floor() as i32 & ((1 << POINT_AMOUNT as i32) - 1)) as usize;
        let j = (p.y.floor() as i32 & ((1 << POINT_AMOUNT as i32) - 1)) as usize;
        let k = (p.z.floor() as i32 & ((1 << POINT_AMOUNT as i32) - 1)) as usize;
        let mut c = [
            [
                [&Vec3::zero(), &Vec3::zero()],
                [&Vec3::zero(), &Vec3::zero()],
            ],
            [
                [&Vec3::zero(), &Vec3::zero()],
                [&Vec3::zero(), &Vec3::zero()],
            ],
        ];
        for di in 0..2 {
            for dj in 0..2 {
                for dk in 0..2 {
                    c[di][dj][dk] = &self.random_vec[self.perm_x[(i + di) & 255]
                        ^ self.perm_y[(j + dj) & 255]
                        ^ self.perm_z[(k + dk) & 255]];
                }
            }
        }

        Self::perlin_interp(c, u, v, w)
    }

    fn turb(&self, p: &Point3, depth: usize) -> Real {
        let mut accum = 0.0;
        let mut temp_p = p.clone();
        let mut weight = 1.0;

        for _ in 0..depth {
            accum += weight * self.noise(&temp_p);
            weight *= 0.5;
            temp_p *= 2.0;
        }

        return accum.abs();
    }

    fn perlin_generate_perm(p: &mut [usize; 1 << POINT_AMOUNT], rng: &mut Random) {
        for (i, value) in p.iter_mut().enumerate() {
            *value = i;
        }

        Self::permute(p, rng);
    }

    fn permute(p: &mut [usize; 1 << POINT_AMOUNT], rng: &mut Random) {
        for i in (1..(1 << POINT_AMOUNT)).rev() {
            let target = rng.random_range(0..i);
            let tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }

    fn perlin_interp(c: [[[&Vec3; 2]; 2]; 2], u: Real, v: Real, w: Real) -> Real {
        let uu = u * u * (3.0 - 2.0 * u);
        let vv = v * v * (3.0 - 2.0 * v);
        let ww = w * w * (3.0 - 2.0 * w);

        let mut accum = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let weight_vec = Vec3::new(u - i as Real, v - j as Real, w - k as Real);
                    accum += (i as Real * uu + (1.0 - i as Real) * (1.0 - uu))
                        * (j as Real * vv + (1.0 - j as Real) * (1.0 - vv))
                        * (k as Real * ww + (1.0 - k as Real) * (1.0 - ww))
                        * c[i][j][k].dot(&weight_vec);
                }
            }
        }

        accum
    }
}
