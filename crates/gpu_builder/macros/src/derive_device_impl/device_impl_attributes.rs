use proc_macro2::Span;
use syn::{parse::Parse, Error, Ident, Lifetime};

pub struct DeviceImplAttributes {
    pub builder: Ident,
    pub lifetime: Option<Lifetime>,
    pub extra_lifetime: Vec<Lifetime>,
}

impl Parse for DeviceImplAttributes {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut builder = None;
        let mut lifetime = None;
        let mut extra_lifetime = Vec::new();

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            if ident == "builder" {
                // skip the "="
                input.parse::<syn::Token![=]>()?;

                builder = Some(input.parse()?);
            } else if ident == "lifetime" {
                // skip the "="
                input.parse::<syn::Token![=]>()?;

                lifetime = Some(input.parse()?);
            } else if ident == "extra_lifetime" {
                // skip the "="
                input.parse::<syn::Token![=]>()?;

                extra_lifetime.push(input.parse()?);
            } else {
                return Err(Error::new_spanned(
                    ident,
                    "Unexpected identifier, expected builder, lifetime or extra_lifetime",
                ));
            }
            if !input.is_empty() {
                input.parse::<syn::Token![,]>()?;
            }
        }
        let builder = builder
            .ok_or_else(|| Error::new(Span::call_site(), "Missing required `builder` attribute"))?;

        Ok(DeviceImplAttributes {
            builder,
            lifetime,
            extra_lifetime,
        })
    }
}
