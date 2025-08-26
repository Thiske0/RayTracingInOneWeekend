use crate::{
    color::Color,
    random::Random,
    textures::{
        checker::CheckerTexture,
        image::{ImageTexture, ImageTextureDevice},
        perlin::{PerlinTexture, PerlinTextureDevice},
        solid::SolidTexture,
    },
    vec3::{Point3, Real},
};

use enum_dispatch::enum_dispatch;
use gpu_builder::Builder;

#[enum_dispatch]
pub trait Texture {
    /// Returns the color of the texture at the given UV coordinates and point.
    fn color(&self, u: Real, v: Real, p: &Point3, rng: &mut Random) -> Color;
}

#[enum_dispatch(Texture)]
#[derive(Builder)]
#[use_lifetime("'b")]
#[repr(C)]
pub enum TextureKind<'b> {
    SolidTexture(SolidTexture),
    CheckerTexture(CheckerTexture),
    ImageTexture(ImageTexture<'b>),
    PerlinTexture(PerlinTexture<'b>),
}

pub mod checker;
pub mod image;
pub mod perlin;
pub mod solid;
