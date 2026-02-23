use syn::{parse::Parse, Lifetime};

pub struct BuilderAttributes {
    pub lifetime: Option<Lifetime>,
}

impl Parse for BuilderAttributes {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let lifetime = if !input.is_empty() {
            let lifetime: Lifetime = input.parse()?;
            Some(lifetime)
        } else {
            None
        };
        Ok(BuilderAttributes { lifetime })
    }
}
