#[cfg(not(target_os = "cuda"))]
use crate::textures::TextureKind;
use crate::{
    color::Color,
    random::Random,
    textures::Texture,
    vec3::{Point3, Real},
};
use core::cmp::{max, min};
use gpu_builder::derive_builder;
use grid_nd::{GridND, GridNDDevice};
use ref_builder::{RefBuilder, RefBuilderDevice};

#[cfg(not(target_os = "cuda"))]
use image::ImageReader;

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Clone))]
#[derive_builder('a)]
pub struct ImageTexture<'a> {
    image: RefBuilder<'a, GridND<'a, Color, 2>>,
}

#[cfg(not(target_os = "cuda"))]
impl<'a> ImageTexture<'a> {
    pub fn new(image: &'a GridND<Color, 2>) -> TextureKind<'a> {
        ImageTexture {
            image: RefBuilder::new(image),
        }
        .into()
    }

    pub fn from_file(path: &str) -> Result<GridND<'a, Color, 2>, Box<dyn std::error::Error>> {
        let image = ImageReader::open(path)?.decode()?;

        let mut image_grid = GridND::<Color, 2>::new(
            [image.height() as usize, image.width() as usize],
            Color::black(),
        );

        for (y, mut row) in (&mut image_grid).into_iter().enumerate() {
            for (x, pixel) in (&mut row).into_iter().enumerate() {
                use image::GenericImageView;

                let rgba = image.get_pixel(x as u32, image.height() - 1 - y as u32).0;
                *pixel = Color::new(
                    rgba[0] as Real / 255.0,
                    rgba[1] as Real / 255.0,
                    rgba[2] as Real / 255.0,
                );
            }
        }
        Ok(image_grid)
    }
}

impl<'a> Texture for ImageTexture<'a> {
    fn color(&self, u: Real, v: Real, _p: &Point3, _rng: &mut Random) -> Color {
        let x = (u * self.image.shape()[1] as Real).floor() as usize;
        let x = min(max(x, 0), self.image.shape()[1] - 1);
        let y = (v * self.image.shape()[0] as Real).floor() as usize;
        let y = min(max(y, 0), self.image.shape()[0] - 1);
        self.image.at(y).at(x).clone()
    }
}
