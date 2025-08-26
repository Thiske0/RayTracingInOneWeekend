use crate::{
    color::Color,
    random::Random,
    textures::Texture,
    vec3::{Point3, Real},
};
use gpu_builder::Builder;
use grid_nd::{GridND, GridNDDevice};
use ref_builder::{RefBuilder, RefBuilderDevice};

#[cfg(not(target_os = "cuda"))]
use image::ImageReader;

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;

#[repr(C)]
#[derive(Builder)]
#[use_lifetime("'a")]
pub struct ImageTexture<'a> {
    image: RefBuilder<'a, GridND<'a, Color, 2>>,
}

#[cfg(not(target_os = "cuda"))]
impl<'a> ImageTexture<'a> {
    pub fn new(image: &'a GridND<Color, 2>) -> Self {
        ImageTexture {
            image: RefBuilder::new(image),
        }
    }

    pub fn from_file(path: &str) -> Result<GridND<'a, Color, 2>, Box<dyn std::error::Error>> {
        let image = ImageReader::open(path)?.decode()?;

        Ok(GridND::<Color, 2>::new(
            [image.height() as usize, image.width() as usize],
            Color::black(),
        ))
    }
}

impl<'a> Texture for ImageTexture<'a> {
    fn color(&self, u: Real, v: Real, _p: &Point3, _rng: &mut Random) -> &Color {
        let x = (u * self.image.shape()[1] as Real).floor() as usize;
        let y = (v * self.image.shape()[0] as Real).floor() as usize;
        self.image.at(y).at(x)
    }
}
