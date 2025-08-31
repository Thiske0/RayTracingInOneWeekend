use std::collections::HashSet;

use proc_macro2::Span;
use quote::{quote, ToTokens};
use syn::{
    parse_quote, punctuated::Punctuated, spanned::Spanned, Attribute, Error, Fields, GenericParam,
    Generics, Ident, Lifetime, LifetimeParam, Result, Token, TraitBound, Type, Variant,
};

pub fn require_repr(attributes: &Vec<Attribute>) -> Result<()> {
    for attr in attributes {
        if attr.path().is_ident("repr") {
            let found = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("C") || meta.path.is_ident("transparent") {
                    Ok(())
                } else {
                    Err(Error::new(
                        meta.path.span(),
                        "expected `C` or `transparent`",
                    ))
                }
            });
            if found.is_ok() {
                return Ok(());
            }
        }
    }

    return Err(Error::new(
        Span::call_site(),
        "#[repr(C)] or #[repr(transparent)] required",
    ));
}

pub fn add_lifetimes(generics: &Generics, lifetimes: Vec<&Lifetime>) -> syn::Generics {
    let mut new_generics = generics.clone();
    for lt in lifetimes.into_iter().rev() {
        new_generics.params.insert(
            0,
            syn::GenericParam::Lifetime(LifetimeParam::new(lt.clone())),
        );
    }
    new_generics
}

pub fn host_generics_from_device(
    device_generics: &Generics,
    builder_lifetime: &Option<Lifetime>,
    extra_lifetimes: &Vec<Lifetime>,
) -> (Generics, proc_macro2::TokenStream) {
    let all_lifetimes = if let Some(builder_lifetime) = builder_lifetime {
        std::iter::once(builder_lifetime)
            .chain(extra_lifetimes.iter())
            .collect::<Vec<_>>()
    } else {
        extra_lifetimes.iter().collect::<Vec<_>>()
    };
    let builder_lifetime = builder_lifetime
        .clone()
        .unwrap_or(Lifetime::new("'_", Span::mixed_site()));

    let mut result = add_lifetimes(device_generics, all_lifetimes);
    let mut changed_types = Vec::<Type>::new();
    result.params.iter_mut().for_each(|param| {
        if let GenericParam::Type(ref mut ty_param) = param {
            let ident = &ty_param.ident;
            for bound in &mut ty_param.bounds {
                if let syn::TypeParamBound::Trait(ref mut trait_bound) = bound {
                    let new_bound = (|| {
                        for path_segment in &mut trait_bound.path.segments {
                            if path_segment.ident == "gpu_builder" {
                                continue;
                            } else if path_segment.ident == "BuildResultType" {
                                if let syn::PathArguments::None = path_segment.arguments {
                                    let param_type = parse_quote! { #ident };
                                    changed_types.push(param_type);
                                    return Some(TraitBound::from(
                                        syn::parse_quote! { gpu_builder::Builder<#builder_lifetime> },
                                    ));
                                } else {
                                    return None;
                                }
                            } else {
                                return None;
                            }
                        }
                        None
                    })();
                    if let Some(new_bound) = new_bound {
                        *trait_bound = new_bound;
                    }
                }
            }
        }
    });

    (
        result,
        make_device_from_host_generics_ty(device_generics, &changed_types),
    )
}

pub fn device_generics_from_host(host_generics: &Generics) -> (Generics, Vec<Type>) {
    let mut result = host_generics.clone();
    let mut changed_types: Vec<Type> = Vec::<Type>::new();
    result.params.iter_mut().for_each(|param| {
        if let GenericParam::Type(ref mut ty_param) = param {
            let ident = &ty_param.ident;
            for bound in &mut ty_param.bounds {
                if let syn::TypeParamBound::Trait(ref mut trait_bound) = bound {
                    let new_bound = (|| {
                        for path_segment in &mut trait_bound.path.segments {
                            if path_segment.ident == "gpu_builder" {
                                continue;
                            } else if path_segment.ident == "Builder" {
                                if let syn::PathArguments::AngleBracketed(ref mut args) =
                                    path_segment.arguments
                                {
                                    if args.args.len() != 1 {
                                        return None;
                                    } else if let syn::GenericArgument::Lifetime(_lifetime) =
                                        &args.args[0]
                                    {
                                        let param_type = parse_quote! { #ident };
                                        changed_types.push(param_type);
                                        return Some(TraitBound::from(
                                            syn::parse_quote! { gpu_builder::BuildResultType },
                                        ));
                                    } else {
                                        return None;
                                    }
                                } else {
                                    return None;
                                }
                            } else {
                                return None;
                            }
                        }
                        None
                    })();
                    if let Some(new_bound) = new_bound {
                        *trait_bound = new_bound;
                    }
                }
            }
        }
    });
    result.params = result
        .params
        .into_iter()
        .filter(|param| {
            if let GenericParam::Lifetime(_lifetime) = param {
                false
            } else {
                true
            }
        })
        .collect();

    (result, changed_types)
}

pub fn device_generics_ty_from_host(host_generics: &Generics) -> proc_macro2::TokenStream {
    let (device_generics, changed_types) = device_generics_from_host(host_generics);
    make_device_from_host_generics_ty(&device_generics, &changed_types)
}

fn make_device_from_host_generics_ty(
    device_generics: &Generics,
    changed_types: &Vec<Type>,
) -> proc_macro2::TokenStream {
    let device_from_host_ty_generics = device_generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Type(ty_param) => {
                let ident = &ty_param.ident;
                let param_type: Type = parse_quote! { #ident };
                let changed_types_tokens = changed_types
                    .iter()
                    .map(|ty| ty.to_token_stream().to_string())
                    .collect::<Vec<_>>();
                if changed_types_tokens.contains(&param_type.to_token_stream().to_string()) {
                    return quote! { #ident::Output };
                }
                quote! { #ident }
            }
            GenericParam::Lifetime(lifetime) => {
                quote! { #lifetime }
            }
            GenericParam::Const(const_param) => {
                let ident = &const_param.ident;
                quote! { #ident }
            }
        })
        .collect::<Vec<_>>();
    quote! { <#(#device_from_host_ty_generics),*> }
}

/// generate_variant is a function that takes the field names and generates the variant body, second arg is function to construct the enum of the variant
pub fn make_variants<F>(
    variants: &Punctuated<Variant, Token![,]>,
    generate_variant: F,
) -> Vec<proc_macro2::TokenStream>
where
    F: Fn(
        Vec<Ident>,
        &dyn Fn(&Ident, Vec<proc_macro2::TokenStream>) -> proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream,
{
    variants
        .iter()
        .map(|variant| {
            let variant_name = &variant.ident;
            match &variant.fields {
                Fields::Named(fields) => {
                    let field_names: Vec<_> = fields
                        .named
                        .iter()
                        .map(|f| f.ident.clone().unwrap())
                        .collect();
                    let body = generate_variant(field_names.clone(), &|result_type: &Ident,
                                                                       fields_result: Vec<
                        proc_macro2::TokenStream,
                    >| {
                        let fields_enum = field_names
                            .iter()
                            .zip(fields_result)
                            .map(|(name, value)| {
                                quote! {
                                    #name: #value,
                                }
                            })
                            .collect::<Vec<_>>();
                        quote! {
                            #result_type::#variant_name {
                                #(#fields_enum)*
                            }
                        }
                    });
                    quote! {
                        Self::#variant_name { #(#field_names),* } => {
                            #body
                        }
                    }
                }
                Fields::Unnamed(fields) => {
                    let field_names: Vec<_> = fields
                        .unnamed
                        .iter()
                        .enumerate()
                        .map(|(i, _f)| Ident::new(&format!("field_{}", i), Span::mixed_site()))
                        .collect();
                    let body = generate_variant(field_names.clone(), &|result_type: &Ident,
                                                                       fields_result: Vec<
                        proc_macro2::TokenStream,
                    >| {
                        quote! {
                            #result_type::#variant_name(#(#fields_result),*)
                        }
                    });
                    quote! {
                        Self::#variant_name(#(#field_names),*) => {
                            #body
                        }
                    }
                }
                Fields::Unit => {
                    let body = generate_variant(vec![], &|result_type: &Ident,
                                                          _fields_result: Vec<
                        proc_macro2::TokenStream,
                    >| {
                        quote! {
                            #result_type::#variant_name
                        }
                    });
                    quote! {
                        Self::#variant_name => {
                            #body
                        }
                    }
                }
            }
        })
        .collect::<Vec<_>>()
}

pub fn generate_fresh_lifetime(generics: &Generics) -> Lifetime {
    let existing: HashSet<String> = generics
        .params
        .iter()
        .filter_map(|p| match p {
            GenericParam::Lifetime(l) => Some(l.lifetime.ident.to_string()),
            _ => None,
        })
        .collect();
    let mut fresh = 1;
    while existing.contains(&format!("'a{}", fresh)) {
        fresh += 1;
    }
    Lifetime::new(&format!("'a{}", fresh), Span::call_site())
}
